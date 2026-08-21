from __future__ import annotations

import ast

from .index import ModInfo, expr_name
from .refs import RefVal

NAMES = {"register", "register_module"}


def reg_maps(info: ModInfo) -> dict[str, dict[str, RefVal]]:
    result: dict[str, dict[str, RefVal]] = {}
    for name, node in info.classes.items():
        for deco in node.decorator_list:
            item = _deco(deco, name)
            if item is None:
                continue
            registry, key = item
            result.setdefault(registry, {})[key] = RefVal(f"{info.name}.{name}")
    return result


def _deco(node: ast.AST, fallback: str) -> tuple[str, str] | None:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in NAMES:
        return None
    registry = expr_name(node.func.value)
    if registry is None:
        return None
    key = _key(node) or fallback
    return registry, key


def _key(node: ast.Call) -> str | None:
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    for item in node.keywords:
        if item.arg in {"name", "module_name"}:
            value = item.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
    return None
