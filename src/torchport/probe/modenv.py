from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .index import ModInfo, canonical
from .refs import FuncVal, RefVal
from .scopes import bind_scope
from .values import MISSING, eval_node


def mod_env(info: ModInfo, modules: dict[str, ModInfo], root: Path) -> dict[str, Any]:
    env: dict[str, Any] = {"__file__": str(info.path)}
    env.update({local: RefVal(canonical(target, modules)) for local, target in info.aliases.items()})
    env.update({name: RefVal(f"{info.name}.{name}") for name in info.classes})
    env.update(info.funcs)
    body = [item for item in info.tree.body if not isinstance(item, (ast.ClassDef, ast.FunctionDef))]
    bind_scope(ast.Module(body=body, type_ignores=[]), env, root, eval_node, MISSING)
    return env


def import_funcs(
    info: ModInfo,
    modules: dict[str, ModInfo],
    funcs: dict[str, tuple[ModInfo, ast.FunctionDef]],
    root: Path,
) -> dict[str, FuncVal]:
    result = {}
    for local, target in info.aliases.items():
        target = canonical(target, modules)
        if target in funcs:
            owner, func = funcs[target]
            result[local] = FuncVal(func, mod_env(owner, modules, root))
        if target in modules:
            owner = modules[target]
            env = mod_env(owner, modules, root)
            for name, func in owner.funcs.items():
                result[f"{local}.{name}"] = FuncVal(func, env)
    return result
