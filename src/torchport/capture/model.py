"""模型入口、框架桥接与权重状态。"""

from __future__ import annotations

import importlib
import runpy
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import file_hash

STATE_KEYS = ("state_dict", "model_state_dict", "model", "net", "weights")
PREFIXES = ("module.", "_orig_mod.", "model.")
SKIP_KEYS = ("num_batches_tracked", "attn_mask")


def import_paths(root: Path) -> list[Path]:
    return [p for p in (root / "src",) if p.is_dir()] + [root]


def module_name(path: Path, root: Path) -> str | None:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts.pop(0)
    if parts and parts[-1] == "__init__":
        parts.pop()
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def load_scope(root: Path, entry: str) -> dict[str, Any]:
    original, argv = list(sys.path), list(sys.argv)
    sys.path[:0] = [str(path) for path in import_paths(root)]
    sys.argv = [str(root / entry)]
    try:
        name = module_name(root / entry, root)
        if name is None:
            return runpy.run_path(str(root / entry), run_name="torchport_probe")
        importlib.invalidate_caches()
        return vars(importlib.import_module(name))
    finally:
        sys.path[:], sys.argv = original, argv


# ---------- 框架适配（source=torch，target=jittor）----------


def _fw(kind: str) -> Any:
    if kind == "source":
        import torch

        return torch
    import jittor

    return jittor


def _tensor_type(kind: str) -> type:
    fw = _fw(kind)
    return fw.Tensor if kind == "source" else fw.Var


def _to_numpy(tensor: Any, kind: str) -> np.ndarray:
    # 必须复制：CPU 张量的 .cpu().numpy() 与张量共享内存，hook 记录后模型里的
    # 原地运算（ReLU(inplace=True)、out += ...）会改写已记录的值。
    if kind == "source":
        return np.array(tensor.detach().cpu().numpy(), copy=True)
    return np.array(tensor.numpy(), copy=True)


def _from_numpy(value: np.ndarray, kind: str) -> Any:
    fw = _fw(kind)
    return fw.from_numpy(np.asarray(value)) if kind == "source" else fw.array(value)


def flatten(value: Any, prefix: str, is_tensor: Callable[[Any], bool], convert: Callable) -> dict:
    """按稳定的容器顺序展平张量叶子。"""
    if is_tensor(value):
        return {prefix: convert(value)}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    else:
        return {}
    result: dict[str, Any] = {}
    for key, item in items:
        result.update(flatten(item, f"{prefix}.{key}", is_tensor, convert))
    return result


def to_arrays(value: Any, kind: str, prefix: str) -> dict[str, np.ndarray]:
    kind_type = _tensor_type(kind)
    return flatten(value, prefix, lambda v: isinstance(v, kind_type), lambda t: _to_numpy(t, kind))


def find_tensor(value: Any, kind: str) -> Any | None:
    arrays = flatten(value, "v", lambda v: isinstance(v, _tensor_type(kind)), lambda t: t)
    return next(iter(arrays.values()), None)


def to_array(value: Any, kind: str) -> np.ndarray:
    arrays = to_arrays(value, kind, "value")
    if not arrays:
        raise ValueError("模型输出中没有张量")
    return next(iter(arrays.values()))


def _encode_call_value(
    value: Any,
    kind: str,
    arrays: dict[str, np.ndarray],
    tensors: dict[int, str],
) -> dict[str, Any]:
    """把一次模块调用编码成 JSON 结构，张量内容单独放进 NPZ。"""
    if isinstance(value, _tensor_type(kind)):
        identity = id(value)
        name = tensors.get(identity)
        if name is None:
            name = f"tensor.{len(arrays)}"
            tensors[identity] = name
            arrays[name] = _to_numpy(value, kind)
        return {"tensor": name}
    if isinstance(value, tuple):
        return {"tuple": [_encode_call_value(item, kind, arrays, tensors) for item in value]}
    if isinstance(value, list):
        return {"list": [_encode_call_value(item, kind, arrays, tensors) for item in value]}
    if isinstance(value, dict):
        return {
            "dict": {
                str(key): _encode_call_value(item, kind, arrays, tensors)
                for key, item in value.items()
            }
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"value": value}
    raise TypeError(f"不支持的模块输入类型：{type(value).__module__}.{type(value).__name__}")


def _decode_call_value(value: dict[str, Any], arrays: Any, kind: str) -> Any:
    if "tensor" in value:
        return _from_numpy(np.asarray(arrays[value["tensor"]]), kind)
    if "tuple" in value:
        return tuple(_decode_call_value(item, arrays, kind) for item in value["tuple"])
    if "list" in value:
        return [_decode_call_value(item, arrays, kind) for item in value["list"]]
    if "dict" in value:
        return {key: _decode_call_value(item, arrays, kind) for key, item in value["dict"].items()}
    return value.get("value")


# ---------- 模型构造 ----------


class Config(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _config(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({key: _config(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return type(value)(_config(item) for item in value)
    return value


def _config_args(args: list[Any], kwargs: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    return [_config(item) for item in args], {key: _config(v) for key, v in kwargs.items()}


def _attr(value: Any, parts: list[str]) -> Any | None:
    for part in parts:
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _lookup(scope: dict[str, Any], name: str) -> Any | None:
    parts = name.split(".")
    candidate = scope.get(parts[0])
    if candidate is not None and (found := _attr(candidate, parts[1:])) is not None:
        return found
    for size in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:size]))
        except ImportError:
            continue
        if (found := _attr(module, parts[size:])) is not None:
            return found
    return candidate if len(parts) == 1 else None


def load_model(scope: dict[str, Any], name: str, kind: str, args: list, kwargs: dict) -> Any:
    base = _fw(kind).nn.Module
    factory = _lookup(scope, name)
    if not callable(factory):
        raise ValueError(f"入口中没有可调用的模型或工厂：{name}")
    try:
        model = factory(*deepcopy(args), **deepcopy(kwargs))
    except AttributeError as exc:
        if "'dict' object has no attribute" not in str(exc):
            raise
        cfg_args, cfg_kwargs = _config_args(args, kwargs)
        model = factory(*cfg_args, **cfg_kwargs)
    except TypeError as exc:
        raise ValueError(f"模型构造参数不匹配：{name}: {exc}") from exc
    if not isinstance(model, base):
        raise ValueError(f"工厂没有返回 {kind} 模型：{name}")
    return model


# ---------- 权重 ----------


def _portable(name: str) -> bool:
    return not name.endswith(SKIP_KEYS)


def _state(value: Any) -> dict[str, Any]:
    if hasattr(value, "state_dict"):
        value = value.state_dict()
    if not isinstance(value, dict):
        raise ValueError("检查点不是状态字典")
    for key in STATE_KEYS:
        item = value.get(key)
        item = item.state_dict() if hasattr(item, "state_dict") else item
        if isinstance(item, dict):
            value = item
            break
    return {str(name): item for name, item in value.items() if hasattr(item, "shape")}


def _matched(weights: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    matched: dict[str, Any] = {}
    for name, value in weights.items():
        if name not in target:
            continue
        source_shape = tuple(value.shape)
        target_shape = tuple(target[name].shape)
        if source_shape == target_shape:
            matched[name] = value
        elif int(np.prod(source_shape)) == int(np.prod(target_shape)) == 1:
            matched[name] = value.reshape(target_shape)
    return matched


def load_checkpoint(model: Any, path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise ValueError(f"源检查点不存在：{path}")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: torch.from_numpy(np.asarray(archive[name])) for name in archive.files}
    elif path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        raw = load_file(str(path), device="cpu")
    else:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    weights = _state(raw)
    target = model.state_dict()
    choices = [weights] + [
        {n[len(p) :] if n.startswith(p) else n: v for n, v in weights.items()} for p in PREFIXES
    ]
    choices.extend(
        {f"model.{name}": value for name, value in items.items()} for items in list(choices)
    )
    selected = max(choices, key=lambda items: len(_matched(items, target)))
    loaded = _matched(selected, target)
    if not loaded:
        raise ValueError(f"检查点没有可加载的模型权重：{path}")
    model.load_state_dict(loaded, strict=False)
    missing = sorted(n for n in set(target) - set(loaded) if _portable(n))
    return {
        "requested": True,
        "path": str(path),
        "sha256": file_hash(path),
        "loaded": sum(_portable(n) for n in loaded),
        "total": sum(_portable(n) for n in target),
        "missing": missing,
        "unexpected": sorted(n for n in set(selected) - set(loaded) if _portable(n)),
        "complete": not missing,
    }


def _storage_key(item: Any) -> tuple | None:
    """共享存储的身份：PyTorch state_dict 对绑定权重给出同一存储的多个视图，Jittor 给出同一 Var。"""
    if hasattr(item, "data_ptr"):
        if item.numel() == 0:
            return None
        return (item.data_ptr(), tuple(item.shape), tuple(item.stride()), str(item.dtype))
    return ("var", id(item))


def unique_state(model: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """可移植状态按存储去重：同一张量只保留首次出现的名字，其余名字记为别名。

    绑定权重（如 GPT-2 的 lm_head 与 wte）在 PyTorch 中列两次、在 Jittor 中列一次；
    按名字计数会把同一参数误判为缺少或多出。拆成两个独立参数的实现仍会多出参数。
    """
    unique: dict[str, Any] = {}
    aliases: dict[str, str] = {}
    owners: dict[tuple, str] = {}
    for name, item in model.state_dict().items():
        if not _portable(name):
            continue
        key = _storage_key(item)
        if key is not None and key in owners:
            aliases[name] = owners[key]
            continue
        if key is not None:
            owners[key] = name
        unique[name] = item
    return unique, aliases


def save_state(model: Any, path: Path) -> int:
    state, _ = unique_state(model)
    np.savez(path, **{name: item.detach().cpu().numpy() for name, item in state.items()})
    return len(state)


def load_state(model: Any, path: Path, kind: str = "target") -> dict[str, Any]:
    """加载 oracle 状态并逐参数回读核验形状、dtype 与数值。"""
    with np.load(path, allow_pickle=False) as archive:
        weights = {n: np.asarray(archive[n]) for n in archive.files if _portable(n)}
    target, aliases = unique_state(model)
    matched = _matched(weights, target)
    if kind == "source":
        import torch

        model.load_state_dict(
            {name: torch.from_numpy(np.asarray(value)) for name, value in matched.items()},
            strict=False,
        )
    else:
        model.load_parameters(matched)
    after = model.state_dict()
    mismatched = [
        name
        for name, value in matched.items()
        if not np.array_equal(np.asarray(_to_numpy(after[name], kind)), value)
    ]
    # target 多出的参数没有 oracle 值，初始化随机，必须进入证据而不是被忽略
    extra = sorted(n for n in set(target) - set(matched) if _portable(n))
    return {
        "total": len(weights),
        "target_total": sum(_portable(n) for n in target),
        "loaded": len(matched) - len(mismatched),
        "missing": sorted(set(weights) - set(matched)),
        "extra": extra,
        "value_mismatch": mismatched,
        # 绑定权重的其余名字：只作证据，不计入参数完整性
        "aliases": aliases,
    }


def _synchronize(kind: str, device: str, result: Any) -> None:
    if kind == "source":
        if device == "cuda":
            _fw(kind).cuda.synchronize()
        return
    tensor = find_tensor(result, kind)
    if tensor is None:
        raise ValueError("模型输出中没有可同步的 Jittor 张量")
    tensor.sync()
