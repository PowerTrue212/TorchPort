from __future__ import annotations

import importlib
import runpy
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .paths import import_paths, module_name
from .walk import flatten


def load_scope(root: Path, entry: str) -> dict[str, Any]:
    original = list(sys.path)
    argv = list(sys.argv)
    sys.path[:0] = [str(path) for path in import_paths(root)]
    sys.argv = [str(root / entry)]
    try:
        name = module_name(root / entry, root)
        if name is None:
            return runpy.run_path(str(root / entry), run_name="torchport_probe")
        importlib.invalidate_caches()
        return vars(importlib.import_module(name))
    finally:
        sys.path[:] = original
        sys.argv = argv


def find_tensor(value: Any, kind: str) -> Any | None:
    if kind == "source":
        import torch

        tensor_type = torch.Tensor
    else:
        import jittor as jt

        tensor_type = jt.Var
    if isinstance(value, tensor_type):
        return value
    if isinstance(value, dict):
        for item in value.values():
            if (result := find_tensor(item, kind)) is not None:
                return result
    if isinstance(value, (list, tuple)):
        for item in value:
            if (result := find_tensor(item, kind)) is not None:
                return result
    return None


def to_array(value: Any, kind: str) -> np.ndarray:
    arrays = to_arrays(value, kind, "value")
    if not arrays:
        raise ValueError("模型输出中没有张量")
    return next(iter(arrays.values()))


def to_arrays(value: Any, kind: str, prefix: str) -> dict[str, np.ndarray]:
    if kind == "source":
        import torch

        tensor_type = torch.Tensor

        def convert(tensor: Any) -> np.ndarray:
            return np.asarray(tensor.detach().cpu().numpy())

    else:
        import jittor as jt

        tensor_type = jt.Var

        def convert(tensor: Any) -> np.ndarray:
            return np.asarray(tensor.numpy())

    return flatten(value, prefix, lambda item: isinstance(item, tensor_type), convert)


def load_model(
    scope: dict[str, Any],
    name: str,
    kind: str,
    args: list[Any],
    kwargs: dict[str, Any],
) -> Any:
    if kind == "source":
        import torch

        base = torch.nn.Module
    else:
        import jittor as jt

        base = jt.nn.Module
    candidate = _lookup(scope, name)
    if candidate is None:
        raise ValueError(f"入口中没有模型或工厂：{name}")
    if not callable(candidate):
        raise ValueError(f"对象不能创建模型：{name}")
    try:
        model = candidate(*deepcopy(args), **deepcopy(kwargs))
    except AttributeError as exc:
        if not _config_miss(exc, args, kwargs):
            raise
        cfg_args, cfg_kwargs = _config_args(args, kwargs)
        model = candidate(*cfg_args, **cfg_kwargs)
    except TypeError as exc:
        raise ValueError(f"模型构造参数不匹配：{name}: {exc}") from exc
    if not isinstance(model, base):
        raise ValueError(f"工厂没有返回 {kind} 模型：{name}")
    return model


def _lookup(scope: dict[str, Any], name: str) -> Any | None:
    parts = name.split(".")
    candidate = scope.get(parts[0])
    rest = parts[1:]
    if candidate is not None:
        found = _attr(candidate, rest)
        if found is not None:
            return found
    for size in range(len(parts) - 1, 0, -1):
        try:
            candidate = importlib.import_module(".".join(parts[:size]))
        except ImportError:
            continue
        found = _attr(candidate, parts[size:])
        if found is not None:
            return found
    return candidate if not rest else None


def _attr(value: Any, parts: list[str]) -> Any | None:
    candidate = value
    for part in parts:
        candidate = getattr(candidate, part, None)
        if candidate is None:
            return None
    return candidate


class Config(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _config_args(args: list[Any], kwargs: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    return [_config(item) for item in args], {
        key: _config(value) for key, value in kwargs.items()
    }


def _config(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({key: _config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_config(item) for item in value)
    return value


def _config_miss(exc: AttributeError, args: list[Any], kwargs: dict[str, Any]) -> bool:
    return "'dict' object has no attribute" in str(exc) and (
        _has_dict(args) or _has_dict(kwargs)
    )


def _has_dict(value: Any) -> bool:
    if isinstance(value, dict):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_dict(item) for item in value)
    return False


def add_hooks(model: Any, kind: str, values: dict[str, np.ndarray]) -> list[Any]:
    counts: Counter[str] = Counter()
    handles = []

    def make(name: str):
        def hook(_module: Any, _inputs: Any, output: Any, *_extra: Any) -> None:
            counts[name] += 1
            key = name if counts[name] == 1 else f"{name}#{counts[name]}"
            values.update(to_arrays(output, kind, key))

        return hook

    for name, module in model.named_modules():
        if name:
            handle = module.register_forward_hook(make(name))
            if handle is not None:
                handles.append(handle)
    return handles


def func_scopes(model: Any, kind: str) -> list[dict[str, Any]]:
    """返回模型方法真实使用的全局字典，供直接导入函数探针使用。"""
    name = "forward" if kind == "source" else "execute"
    scopes = []
    seen = set()
    for _path, module in model.named_modules():
        method = getattr(module, name, None)
        func = getattr(method, "__func__", method)
        scope = getattr(func, "__globals__", None)
        if isinstance(scope, dict) and id(scope) not in seen:
            seen.add(id(scope))
            scopes.append(scope)
    return scopes
