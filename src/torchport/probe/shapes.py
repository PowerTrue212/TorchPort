from __future__ import annotations

import ast
from typing import Any


def input_specs(model: ast.ClassDef, values: dict[str, Any]) -> list[dict[str, Any]]:
    for node in ast.walk(model):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if name == "Linear" and node.args:
            size = _value(node.args[0], values)
            return [_spec([1, size])] if size else []
        dims = {"Conv1d": 1, "Conv2d": 2, "Conv3d": 3}
        if name in dims and node.args:
            channels = _value(node.args[0], values)
            if channels:
                return [_spec([1, channels, *([32] * dims[name])])]
    return []


def _value(node: ast.AST, values: dict[str, Any]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name) and isinstance(values.get(node.id), int):
        return values[node.id]
    return None


def _spec(shape: list[int]) -> dict[str, Any]:
    return {"shape": shape, "dtype": "float32"}
