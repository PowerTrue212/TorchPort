"""单端采集：加载模型、输入、权重，正常执行并保存最终输出；trace 为显式诊断模式。

在子进程内运行（`python -m torchport _capture <side.json>`），只导入本端框架。
"""

from __future__ import annotations

import importlib
import inspect
import json
import runpy
import sys
import time
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .execution import read_json, write_json
from .task import file_hash

MODES = ("normal", "trace")
OP_NAMES = (
    "relu",
    "gelu",
    "silu",
    "softmax",
    "log_softmax",
    "max_pool2d",
    "avg_pool2d",
    "adaptive_avg_pool2d",
    "interpolate",
    "normalize",
    "pad",
)
RAND_NAMES = ("rand", "randn")
STATE_KEYS = ("state_dict", "model_state_dict", "model", "net", "weights")
PREFIXES = ("module.", "_orig_mod.", "model.")
SKIP_KEYS = ("num_batches_tracked", "attn_mask")


# ---------- 路径与入口 ----------


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
    if kind == "source":
        return np.asarray(tensor.detach().cpu().numpy())
    return np.asarray(tensor.numpy())


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


def save_state(model: Any, path: Path) -> int:
    weights = {
        name: item.detach().cpu().numpy()
        for name, item in model.state_dict().items()
        if _portable(name)
    }
    np.savez(path, **weights)
    return len(weights)


def load_state(model: Any, path: Path) -> dict[str, Any]:
    """加载 oracle 状态并逐参数回读核验形状、dtype 与数值。"""
    with np.load(path, allow_pickle=False) as archive:
        weights = {n: np.asarray(archive[n]) for n in archive.files if _portable(n)}
    target = model.state_dict()
    matched = _matched(weights, target)
    model.load_parameters(matched)
    after = model.state_dict()
    mismatched = [
        name
        for name, value in matched.items()
        if not np.array_equal(np.asarray(_to_numpy(after[name], "target")), value)
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
    }


# ---------- 输入 ----------


def input_key(index: int, spec: dict[str, Any]) -> str:
    arg, key = spec.get("arg"), spec.get("key")
    if spec.get("pack"):
        return f"input.{arg}.{spec['item']}" if arg else f"input.item.{spec['item']}"
    if key:
        return f"input.{arg}.{key}" if arg else f"input.{key}"
    if arg:
        return f"input.{arg}"
    return "input" if index == 0 else f"input.{index}"


def make_data(inputs: list[dict[str, Any]], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = {}
    for index, spec in enumerate(inputs):
        shape = tuple(spec["shape"])
        dtype = str(spec.get("dtype", "float32"))
        if spec.get("path"):
            value = np.load(spec["path"], allow_pickle=False)
            if not isinstance(value, np.ndarray):
                raise ValueError(f"固定输入不是数组：{spec['path']}")
            if tuple(value.shape) != shape or str(value.dtype) != dtype:
                raise ValueError(
                    f"固定输入 shape/dtype 不符：{spec['path']}，"
                    f"期望 {shape}/{dtype}，实际 {value.shape}/{value.dtype}"
                )
        elif dtype == "int64":
            value = rng.integers(0, 8, size=shape, dtype=np.int64)
        else:
            value = rng.standard_normal(shape).astype(np.float32)
        values[input_key(index, spec)] = value
    return values


def pack_values(values: list[Any], inputs: list[dict[str, Any]]) -> tuple[list, dict]:
    """按输入描述组装位置参数、关键字参数、字典和序列。"""
    sequences: dict[str | None, list[Any]] = {}
    kinds: dict[str | None, str] = {}
    for spec, value in zip(inputs, values):
        if spec.get("pack"):
            items = sequences.setdefault(spec.get("arg"), [])
            items.extend([None] * (spec["item"] + 1 - len(items)))
            items[spec["item"]] = value
            kinds[spec.get("arg")] = spec["pack"]
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    pos_dict: dict[str, Any] | None = None
    emitted: set[str | None] = set()
    for spec, value in zip(inputs, values):
        arg, key = spec.get("arg"), spec.get("key")
        if spec.get("pack"):
            if arg in emitted:
                continue
            emitted.add(arg)
            packed = sequences[arg]
            packed = tuple(packed) if kinds[arg] == "tuple" else packed
            (kwargs.__setitem__(arg, packed) if arg else args.append(packed))
        elif key and arg:
            kwargs.setdefault(arg, {})[key] = value
        elif key:
            if pos_dict is None:
                pos_dict = {}
                args.append(pos_dict)
            pos_dict[key] = value
        elif arg:
            kwargs[arg] = value
        else:
            args.append(value)
    return args, kwargs


# ---------- 算子追踪 ----------


def _library_call() -> bool:
    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame and frame.f_back else None
    filename = caller.f_code.co_filename.replace("\\", "/") if caller else ""
    del frame
    return "/site-packages/torch/" in filename or "/site-packages/jittor/" in filename


class OpTrace:
    """临时包装常见函数式算子并记录真实输出。"""

    def __init__(
        self,
        kind: str,
        values: dict[str, np.ndarray],
        scopes: list[dict[str, Any]] | None = None,
    ) -> None:
        self.kind, self.values = kind, values
        self.scopes = scopes or []
        self.counts: Counter[str] = Counter()
        self.saved: list[tuple[Any, str, Callable]] = []
        self.locals: list[tuple[dict, str, Callable]] = []
        self.modules = self._modules(kind)

    @staticmethod
    def _modules(kind: str) -> list[tuple[Any, tuple[str, ...]]]:
        fw = _fw(kind)
        ops = fw.nn.functional if kind == "source" else fw.nn
        return [(ops, OP_NAMES), (fw, RAND_NAMES)]

    def start(self) -> None:
        for module, names in self.modules:
            for name in names:
                func = getattr(module, name, None)
                if not callable(func):
                    continue
                self.saved.append((module, name, func))
                wrapped = self._wrap(name, func)
                setattr(module, name, wrapped)
                for scope in self.scopes:
                    for local, value in list(scope.items()):
                        if value is func:
                            self.locals.append((scope, local, func))
                            scope[local] = wrapped

    def stop(self) -> None:
        for module, name, func in self.saved:
            setattr(module, name, func)
        for scope, name, func in self.locals:
            scope[name] = func
        self.saved, self.locals = [], []

    def _wrap(self, name: str, func: Callable) -> Callable:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _library_call():
                return func(*args, **kwargs)
            self.counts[name] += 1
            count = self.counts[name]
            key = f"op.{name}" if count == 1 else f"op.{name}#{count}"
            output = func(*args, **kwargs)
            tensor = find_tensor(output, self.kind)
            if tensor is not None:
                self.values[key] = to_array(tensor, self.kind)
            return output

        return wrapped


def _defined_in_repo(module: Any, root: Path) -> bool:
    try:
        path = Path(inspect.getsourcefile(type(module)) or "").resolve()
        path.relative_to(root.resolve())
        return True
    except (OSError, TypeError, ValueError):
        return False


def _has_local_state(module: Any) -> bool:
    try:
        return bool(module.state_dict(recurse=False))
    except TypeError:
        # Jittor 1.3 的 state_dict 没有 recurse 参数；局部输入只在 source 端采集。
        return False


def add_hooks(
    model: Any,
    kind: str,
    values: dict[str, np.ndarray],
    *,
    root: Path | None = None,
    probe_values: dict[str, np.ndarray] | None = None,
    probe_calls: list[dict[str, Any]] | None = None,
    probe_inputs: dict[str, np.ndarray] | None = None,
) -> list[Any]:
    counts: Counter[str] = Counter()
    handles = []

    def make(name: str, selected: bool) -> Callable:
        def record(_module: Any, inputs: Any, kwargs: dict[str, Any], output: Any) -> None:
            counts[name] += 1
            key = name if counts[name] == 1 else f"{name}#{counts[name]}"
            output_arrays = to_arrays(output, kind, key)
            values.update(output_arrays)
            if not selected or probe_values is None or probe_calls is None or probe_inputs is None:
                return
            try:
                # 只在一次调用内部去重。跨调用缓存对象 id 会遇到 Python 复用已释放 id，
                # 从而把后续模块错误地指向较早的张量。
                tensor_inputs: dict[int, str] = {}
                call = {
                    "key": key,
                    "module": name,
                    "args": _encode_call_value(inputs, kind, probe_inputs, tensor_inputs),
                    "kwargs": _encode_call_value(kwargs, kind, probe_inputs, tensor_inputs),
                }
            except TypeError as exc:
                probe_calls.append({"key": key, "module": name, "unsupported": str(exc)})
                return
            probe_calls.append(call)
            probe_values.update(output_arrays)

        if kind == "source":
            def hook(_module: Any, inputs: Any, kwargs: dict[str, Any], output: Any) -> None:
                record(_module, inputs, kwargs, output)
        else:
            def hook(_module: Any, inputs: Any, output: Any, *_extra: Any) -> None:
                record(_module, inputs, {}, output)

        return hook

    for name, module in model.named_modules():
        if not name:
            continue
        selected = kind == "source" and root is not None and (
            _defined_in_repo(module, root) or _has_local_state(module)
        )
        hook = make(name, selected)
        if kind == "source":
            handle = module.register_forward_hook(hook, with_kwargs=True)
        else:
            handle = module.register_forward_hook(hook)
        if handle is not None:
            handles.append(handle)
    return handles


def _mark_leaves(calls: list[dict[str, Any]]) -> None:
    """叶子 = 没有任何其他被采集模块位于其名字之下；只有叶子参与噪声标定判定。"""
    names = {call["module"] for call in calls}
    for call in calls:
        prefix = call["module"] + "."
        call["leaf"] = not any(other.startswith(prefix) for other in names)


def _to_float64(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.double() if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(_to_float64(item) for item in value)
    if isinstance(value, list):
        return [_to_float64(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_float64(item) for key, item in value.items()}
    return value


def calibrate_source_modules(
    model: Any,
    calls: list[dict[str, Any]],
    inputs: dict[str, np.ndarray],
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """source 端噪声底：同一份局部输入分别跑 float64 真值和另一条 FP32 执行路径。"""
    import torch

    modules = dict(model.named_modules())
    fp64: dict[str, np.ndarray] = {}
    alt: dict[str, np.ndarray] = {}
    failed: list[dict[str, str]] = []

    def decode(call: dict[str, Any]) -> tuple[Any, Any]:
        args = _decode_call_value(call["args"], inputs, "source")
        kwargs = _decode_call_value(call["kwargs"], inputs, "source")
        if device == "cuda":
            args = _move(args, "cuda")
            kwargs = _move(kwargs, "cuda")
        return args, kwargs

    backends = torch.backends
    saved = (
        backends.cudnn.benchmark,
        backends.cudnn.deterministic,
        backends.cudnn.allow_tf32,
        backends.cuda.matmul.allow_tf32,
    )
    with torch.no_grad():
        for call in calls:
            if call.get("unsupported") or call["module"] not in modules:
                continue
            key, module = call["key"], modules[call["module"]]
            try:
                args, kwargs = decode(call)
                # 另一条 FP32 路径：换 cuDNN 算法选择，TF32 保持关闭，只量化“合法的 FP32 差异”。
                backends.cudnn.benchmark, backends.cudnn.deterministic = True, False
                backends.cudnn.allow_tf32 = backends.cuda.matmul.allow_tf32 = False
                alt.update(to_arrays(module(*args, **kwargs), "source", key))
                (
                    backends.cudnn.benchmark,
                    backends.cudnn.deterministic,
                    backends.cudnn.allow_tf32,
                    backends.cuda.matmul.allow_tf32,
                ) = saved
                double = deepcopy(module).double()
                fp64.update(to_arrays(double(*_to_float64(args), **_to_float64(kwargs)), "source", key))
                del double
            except Exception as exc:  # noqa: BLE001 - 标定失败只让该模块失去判定依据
                failed.append({"key": key, "reason": f"{type(exc).__name__}: {exc}"[:500]})
            finally:
                (
                    backends.cudnn.benchmark,
                    backends.cudnn.deterministic,
                    backends.cudnn.allow_tf32,
                    backends.cuda.matmul.allow_tf32,
                ) = saved
    if device == "cuda":
        torch.cuda.synchronize()
    return fp64, alt, {"fp64_tensors": len(fp64), "alt_tensors": len(alt), "failed": failed}


def _move(value: Any, device: str) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def run_isolated_modules(
    model: Any,
    oracle: Path,
    kind: str,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """用 source 的真实局部输入逐个运行同名 target 模块。"""
    calls_path = oracle / "module_calls.json"
    inputs_path = oracle / "module_inputs.npz"
    if not calls_path.is_file() or not inputs_path.is_file():
        return {}, {"available": False, "reason": "source 没有模块输入证据"}
    calls = json.loads(calls_path.read_text("utf-8"))
    modules = dict(model.named_modules())
    values: dict[str, np.ndarray] = {}
    missing: list[str] = []
    unsupported: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    succeeded = 0
    with np.load(inputs_path, allow_pickle=False) as arrays:
        for call in calls:
            key, name = call["key"], call["module"]
            if reason := call.get("unsupported"):
                unsupported.append({"key": key, "reason": str(reason)})
                continue
            module = modules.get(name)
            if module is None:
                missing.append(key)
                continue
            try:
                args = _decode_call_value(call["args"], arrays, kind)
                kwargs = _decode_call_value(call["kwargs"], arrays, kind)
                output = module(*args, **kwargs)
                _synchronize(kind, device, output)
                values.update(to_arrays(output, kind, key))
                succeeded += 1
            except Exception as exc:  # noqa: BLE001 - 每个模块单独留下可审计失败
                failed.append({"key": key, "reason": f"{type(exc).__name__}: {exc}"[:500]})
    return values, {
        "available": True,
        "recorded": len(calls),
        "succeeded": succeeded,
        "missing_modules": missing,
        "unsupported_inputs": unsupported,
        "failed_modules": failed,
    }


def func_scopes(model: Any, kind: str) -> list[dict[str, Any]]:
    name = "forward" if kind == "source" else "execute"
    scopes, seen = [], set()
    for _path, module in model.named_modules():
        method = getattr(module, name, None)
        scope = getattr(getattr(method, "__func__", method), "__globals__", None)
        if isinstance(scope, dict) and id(scope) not in seen:
            seen.add(id(scope))
            scopes.append(scope)
    return scopes


# ---------- 主流程 ----------


def _digest(values: dict[str, np.ndarray]) -> str:
    import hashlib

    sha = hashlib.sha256()
    for name in sorted(values):
        data = np.ascontiguousarray(values[name])
        sha.update(name.encode() + str(data.dtype).encode() + str(data.shape).encode())
        sha.update(data.tobytes())
    return sha.hexdigest()


def _check_bridge(kind: str) -> None:
    if kind != "target":
        return
    import jittor as jt

    expected = np.asarray([-1.25, 0.0, 2.5], dtype=np.float32)
    if not np.array_equal(expected, np.asarray(jt.array(expected).numpy())):
        raise ValueError(f"Jittor/NumPy 数组桥接不兼容：NumPy {np.__version__}；请使用 NumPy < 2")


def _framework(kind: str, seed: int, device: str) -> dict[str, Any]:
    """设置随机种子与设备，并记录实际生效的设备与缓存路径（不只记录请求值）。"""
    fw = _fw(kind)
    if kind == "source":
        fw.manual_seed(seed)
        if device == "cuda" and not fw.cuda.is_available():
            raise ValueError("task 要求 cuda，但 PyTorch 未检测到可用 GPU")
        return {
            "name": "pytorch",
            "version": fw.__version__,
            "device": device,
            "cuda": fw.version.cuda,
            "device_name": fw.cuda.get_device_name(0) if device == "cuda" else "cpu",
        }
    fw.set_global_seed(seed)
    if device == "cuda":
        if not fw.has_cuda:
            raise ValueError("task 要求 cuda，但 Jittor 未检测到可用 GPU")
        fw.flags.use_cuda = 1
    else:
        fw.flags.use_cuda = 0
    from jittor import compiler

    return {
        "name": "jittor",
        "version": fw.__version__,
        "device": "cuda" if fw.flags.use_cuda else "cpu",
        "cache_path": compiler.cache_path,  # 实际生效路径，供隔离核验
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


def _call_model(model: Any, args: list[Any], kwargs: dict[str, Any], kind: str) -> Any:
    if kind == "source":
        with _fw(kind).no_grad():
            return model(*args, **kwargs)
    return model(*args, **kwargs)


def capture(job: dict[str, Any]) -> dict[str, Any]:
    """job: kind/side/inputs/seed/mode/out_dir，以及 oracle（target 端必需）或 weights。"""
    capture_started = time.time()
    kind, side, mode = job["kind"], job["side"], job.get("mode", "normal")
    if mode not in MODES:
        raise ValueError(f"未知采集模式：{mode}")
    out = Path(job["out_dir"])
    root = Path(side["repo"])
    device = side.get("device", "cpu")
    framework = _framework(kind, job["seed"], device)
    _check_bridge(kind)
    original = list(sys.path)
    sys.path[:0] = [str(path) for path in import_paths(root)]
    try:
        scope = load_scope(root, side["entry"])
        model = load_model(scope, side["model"], kind, side["args"], side["kwargs"])
    finally:
        sys.path[:] = original
    model.eval()
    if kind == "source" and device == "cuda":
        model.cuda()

    if kind == "source":
        weights = job.get("weights")
        checkpoint = (
            load_checkpoint(model, Path(weights))
            if weights
            else {"requested": False, "complete": True}
        )
        arrays = make_data(job["inputs"], job["seed"])
        total = save_state(model, out / "state.npz")
        state = {"total": total, "loaded": total, "missing": [], "value_mismatch": []}
    else:
        oracle = Path(job["oracle"])
        with np.load(oracle / "inputs.npz", allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        state = load_state(model, oracle / "state.npz")
        checkpoint = None

    tensors = [_from_numpy(arrays[input_key(i, s)], kind) for i, s in enumerate(job["inputs"])]
    if kind == "source" and device == "cuda":
        tensors = [tensor.cuda() for tensor in tensors]
    args, kwargs = pack_values(tensors, job["inputs"])
    trace_values: dict[str, np.ndarray] = {}
    module_values: dict[str, np.ndarray] = {}
    module_calls: list[dict[str, Any]] = []
    module_inputs: dict[str, np.ndarray] = {}
    handles = (
        add_hooks(
            model,
            kind,
            trace_values,
            root=root,
            probe_values=module_values,
            probe_calls=module_calls,
            probe_inputs=module_inputs,
        )
        if mode != "normal"
        else []
    )
    ops = (
        OpTrace(kind, trace_values, [scope, *func_scopes(model, kind)])
        if mode != "normal"
        else None
    )
    if ops:
        ops.start()
    try:
        result = _call_model(model, args, kwargs, kind)
        _synchronize(kind, device, result)
        first_output_at = time.time()
    finally:
        if ops:
            ops.stop()
        for handle in handles:
            handle.remove()
    isolated_values: dict[str, np.ndarray] = {}
    isolated = None
    if mode != "normal" and kind == "target":
        isolated_values, isolated = run_isolated_modules(model, Path(job["oracle"]), kind, device)
    performance = None
    if requested := job.get("performance"):
        if mode != "normal":
            raise ValueError("性能测量只允许 mode=normal")
        warmup = int(requested.get("warmup", 10))
        iterations = int(requested.get("iterations", 30))
        if warmup < 0 or iterations < 1:
            raise ValueError("performance warmup 必须非负且 iterations 必须为正数")
        for _ in range(warmup):
            result = _call_model(model, args, kwargs, kind)
        _synchronize(kind, device, result)
        latency_ms = []
        for _ in range(iterations):
            started = time.perf_counter()
            result = _call_model(model, args, kwargs, kind)
            _synchronize(kind, device, result)
            latency_ms.append((time.perf_counter() - started) * 1000)
        ordered = sorted(latency_ms)
        median = float(np.median(ordered))
        p90 = float(ordered[max(0, int(np.ceil(0.9 * len(ordered))) - 1)])
        shape = list(job["inputs"][0].get("shape") or [])
        batch = int(shape[0]) if shape else 1
        performance = {
            "warmup": warmup,
            "iterations": iterations,
            "batch_size": batch,
            "first_output_at": first_output_at,
            "capture_started_at": capture_started,
            "latency_ms": latency_ms,
            "steady_latency_ms": {"median": median, "p90": p90},
            "throughput_per_second": batch / (median / 1000),
        }
    outputs = to_arrays(result, kind, "output")
    if not outputs:
        raise ValueError("模型输出中没有张量")
    if kind == "source":
        np.savez(out / "inputs.npz", **arrays)
    np.savez(out / "outputs.npz", **arrays, **outputs)
    if trace_values:
        np.savez(out / "trace.npz", **trace_values)
    calibration = None
    if kind == "source" and mode != "normal":
        _mark_leaves(module_calls)
        fp64_values, alt_values, calibration = calibrate_source_modules(
            model, module_calls, module_inputs, device
        )
        np.savez(out / "module_fp64.npz", **fp64_values)
        np.savez(out / "module_alt.npz", **alt_values)
        np.savez(out / "module_outputs.npz", **module_values)
        np.savez(out / "module_inputs.npz", **module_inputs)
        (out / "module_calls.json").write_text(
            json.dumps(module_calls, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if isolated_values:
        np.savez(out / "isolated.npz", **isolated_values)
    report = {
        "kind": kind,
        "mode": mode,
        "framework": framework,
        "numpy": np.__version__,
        "weights": state["total"],
        "loaded": state["loaded"],
        "state": state,
        "checkpoint": checkpoint,
        "outputs": sorted(outputs),
        "hashes": {
            "input": _digest(arrays),
            "output": _digest(outputs),
            "state": file_hash(out / "state.npz") if kind == "source" else None,
        },
    }
    if kind == "source" and mode != "normal":
        report["module_probe"] = {
            "calls": len(module_calls),
            "input_tensors": len(module_inputs),
            "output_tensors": len(module_values),
            "calibration": calibration,
        }
    if isolated is not None:
        report["isolated"] = isolated
    if performance is not None:
        report["performance"] = performance
    write_json(out / "capture.json", report)
    return report


def main(job_path: str) -> int:
    capture(read_json(Path(job_path)))
    return 0
