from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from .index import ModInfo
from .scopes import bind_scope
from .values import MISSING, eval_node


def param_defaults(
    func: ast.FunctionDef | None,
    root: Path,
    module: ModInfo,
    skip: int = 0,
) -> dict[str, Any]:
    if func is None:
        return {}
    names = [item.arg for item in func.args.args[skip:]]
    defaults = [None] * (len(names) - len(func.args.defaults))
    defaults.extend(func.args.defaults)
    env = _module_env(root, module)
    result = {}
    for name, node in zip(names, defaults):
        if node is None:
            continue
        value = eval_node(node, env, root)
        if _clean(value):
            result[name] = value
    return result


def param_hints(root: Path, module: ModInfo, names: list[str]) -> dict[str, Any]:
    if not names:
        return {}
    pool = _values(root, module)
    result = {}
    for name in names:
        for key in (name, name.upper(), name.lower()):
            if key in pool and _clean(pool[key]):
                result[name] = pool[key]
                break
    return result


def _values(root: Path, module: ModInfo) -> dict[str, Any]:
    result = {}
    env = _module_env(root, module)
    for name, value in env.items():
        if _clean(value):
            result[name] = value
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and _clean(item):
                    result.setdefault(key, item)
    for name, value in _class_attrs(root, module, env).items():
        result.setdefault(name, value)
    return result


def _module_env(root: Path, module: ModInfo) -> dict[str, Any]:
    env: dict[str, Any] = {"__file__": str(module.path), **module.funcs}
    body = [
        item
        for item in module.tree.body
        if not isinstance(item, (ast.ClassDef, ast.FunctionDef))
    ]
    tree = ast.Module(body=body, type_ignores=[])
    bind_scope(tree, env, root, eval_node, MISSING)
    return env


def _class_attrs(root: Path, module: ModInfo, env: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for owner in module.tree.body:
        if not isinstance(owner, ast.ClassDef):
            continue
        local = dict(env)
        for node in owner.body:
            value_node = None
            targets = []
            if isinstance(node, ast.Assign):
                value_node = node.value
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                value_node = node.value
                targets = [node.target]
            if value_node is None:
                continue
            value = eval_node(value_node, local, root)
            if not _clean(value):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    local[target.id] = value
                    result.setdefault(target.id, value)
    return result


def _clean(value: Any) -> bool:
    if value is MISSING or callable(value) or isinstance(value, (ast.AST, Path)):
        return False
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True
