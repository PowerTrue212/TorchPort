from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .finder import find_model, model_choices
from .shapes import input_specs


def _number(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _shape(node: ast.AST) -> list[int] | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_shape(item) for item in node.elts]
        if not values:
            return [0]
        if any(value is None or value != values[0] for value in values):
            return None
        return [len(values), *values[0]]
    return [] if isinstance(node, ast.Constant) else None


def _literals(tree: ast.AST) -> list[list[int]]:
    names = {"tensor", "zeros", "ones", "rand", "randn"}
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if name not in names:
            continue
        first = node.args[0]
        if name == "tensor":
            shape = _shape(first)
        elif isinstance(first, (ast.List, ast.Tuple)):
            shape = [_number(item) for item in first.elts]
            shape = shape if all(value is not None for value in shape) else None
        else:
            shape = [_number(item) for item in node.args]
            shape = shape if all(value is not None for value in shape) else None
        if shape:
            result.append([int(value) for value in shape])
    return result


def _dataset(tree: ast.AST) -> list[int] | None:
    shapes = {
        "MNIST": [1, 1, 28, 28],
        "FashionMNIST": [1, 1, 28, 28],
        "CIFAR10": [1, 3, 32, 32],
        "CIFAR100": [1, 3, 32, 32],
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in shapes:
            return shapes[node.func.attr]
    return None


def _input(model: ast.ClassDef) -> list[int] | None:
    for node in ast.walk(model):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if name == "Linear" and node.args:
            size = _number(node.args[0])
            return [1, size] if size else None
        dims = {"Conv1d": 1, "Conv2d": 2, "Conv3d": 3}
        if name in dims and node.args:
            channels = _number(node.args[0])
            return [1, channels, *([32] * dims[name])] if channels else None
    return None


def _subs(forward: ast.FunctionDef, name: str) -> list[str | int]:
    result = []
    for node in ast.walk(forward):
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
            continue
        if node.value.id != name:
            continue
        value = node.slice.value if isinstance(node.slice, ast.Constant) else None
        if isinstance(value, (str, int)) and not isinstance(value, bool) and value not in result:
            result.append(value)
    return result


def _pack(arg: ast.arg) -> str | None:
    node = arg.annotation
    if isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute):
        name = node.attr
    elif isinstance(node, ast.Name):
        name = node.id
    else:
        return None
    name = name.lower()
    return name if name in {"list", "tuple"} else None


def _params(model: ast.ClassDef) -> list[dict[str, Any]]:
    forward = next(
        (item for item in model.body if isinstance(item, ast.FunctionDef) and item.name == "forward"),
        None,
    )
    if forward is None:
        return [{}]
    result = []
    for arg in forward.args.args[1:]:
        subs = _subs(forward, arg.arg)
        keys = [item for item in subs if isinstance(item, str)]
        slots = [item for item in subs if isinstance(item, int)]
        pack = _pack(arg)
        if keys:
            result.extend({"key": key} for key in keys)
        elif slots and pack:
            result.extend({"pack": pack, "item": item} for item in slots)
        else:
            result.append({})
    for arg in forward.args.kwonlyargs:
        subs = _subs(forward, arg.arg)
        keys = [item for item in subs if isinstance(item, str)]
        slots = [item for item in subs if isinstance(item, int)]
        pack = _pack(arg)
        if keys:
            result.extend({"arg": arg.arg, "key": key} for key in keys)
        elif slots and pack:
            result.extend({"arg": arg.arg, "pack": pack, "item": item} for item in slots)
        else:
            result.append({"arg": arg.arg})
    return result or [{}]


def _values(model: ast.ClassDef, args: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    result = dict(kwargs)
    init = next(
        (item for item in model.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"),
        None,
    )
    if init is None:
        return result
    names = [item.arg for item in init.args.args[1:]]
    for name, value in zip(names, args):
        result.setdefault(name, value)
    return result


def infer_probe(
    root: Path,
    entry: str | None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    if entry is None:
        return {
            "status": "missing",
            "reason": "没有运行入口",
            "candidates": model_choices(root),
        }
    ref = find_model(root, entry, env)
    if ref is None:
        return {
            "status": "missing",
            "reason": "仓库中没有可访问的模型或工厂",
            "candidates": model_choices(root, entry),
        }
    params = _params(ref.model)
    count = len(params)
    shapes = _literals(ref.tree)[:count]
    dataset = _dataset(ref.tree)
    if len(shapes) < count and dataset:
        shapes.extend([dataset] * (count - len(shapes)))
    specs = input_specs(ref.model, _values(ref.model, ref.model_args, ref.model_kwargs))
    if len(shapes) < count and specs:
        shapes.extend(item["shape"] for item in specs[: count - len(shapes)])
    fallback = _input(ref.model)
    if len(shapes) < count and fallback:
        shapes.extend([fallback] * (count - len(shapes)))
    inputs = [
        {**param, "shape": shape, "dtype": "float32"}
        for param, shape in zip(params, shapes)
    ]
    status = (
        "needs-args"
        if ref.required
        else ("ready" if len(inputs) == count else "needs-input")
    )
    result = {
        "status": status,
        "entry": ref.entry,
        "model": ref.name,
        "inputs": inputs,
        "shape": inputs[0]["shape"] if len(inputs) == 1 else None,
        "dtype": "float32",
        "args": ref.args,
        "kwargs": ref.kwargs,
        "required": ref.required,
        "env": ref.env,
    }
    if status != "ready":
        result["candidates"] = model_choices(root, ref.entry)
    return result
