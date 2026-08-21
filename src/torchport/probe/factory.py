from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from typing import Any

from .index import ModInfo, canonical, expr_name, resolve
from .refs import RefVal
from .registry import reg_maps
from .scopes import bind_scope
from .values import MISSING, call_args, eval_node


def factory(
    info: ModInfo,
    func: ast.FunctionDef,
    modules: dict[str, ModInfo],
    classes: dict[str, tuple[ModInfo, str, ast.ClassDef]],
    root: Path | None = None,
    values: tuple[list[Any], dict[str, Any]] | None = None,
    env: dict[str, Any] | None = None,
) -> ast.ClassDef | None:
    ret = _ret_call(func)
    if ret is None:
        return None
    name = expr_name(ret.func)
    if name and (target := classes.get(resolve(info, name, modules))):
        return target[2]
    if root is None or values is None:
        return None
    local = _local(info, func, values, env or {}, modules, root)
    value = eval_node(ret.func, local, root)
    if isinstance(value, RefVal):
        target = classes.get(canonical(value.name, modules))
        if target:
            return target[2]
    return None


def model_args(
    info: ModInfo,
    func: ast.FunctionDef,
    call: ast.Call,
    env: dict[str, Any],
    root: Path,
    modules: dict[str, ModInfo],
) -> tuple[list[Any], dict[str, Any]] | None:
    values = call_args(call, env, root)
    if values is None:
        return None
    return return_args(info, func, values, env, root, modules)


def return_args(
    info: ModInfo,
    func: ast.FunctionDef,
    values: tuple[list[Any], dict[str, Any]],
    env: dict[str, Any],
    root: Path,
    modules: dict[str, ModInfo],
) -> tuple[list[Any], dict[str, Any]] | None:
    ret = _ret_call(func)
    if ret is None:
        return None
    local = _local(info, func, values, env, modules, root)
    return call_args(ret, local, root)


def external(target: str | None, owner: ModInfo, info: ModInfo) -> bool:
    if target is None or owner is info:
        return False
    return target.endswith((".create_model", ".get_model"))


def _ret_call(func: ast.FunctionDef) -> ast.Call | None:
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            return node.value
    return None


def _local(
    info: ModInfo,
    func: ast.FunctionDef,
    values: tuple[list[Any], dict[str, Any]],
    env: dict[str, Any],
    modules: dict[str, ModInfo],
    root: Path,
) -> dict[str, Any]:
    local = _module_env(info, modules, root)
    local.update(env)
    args, kwargs = values
    params = func.args.args
    defaults = [MISSING] * (len(params) - len(func.args.defaults))
    defaults.extend(eval_node(item, local, root) for item in func.args.defaults)
    for index, param in enumerate(params):
        if index < len(args):
            local[param.arg] = deepcopy(args[index])
        elif param.arg in kwargs:
            local[param.arg] = deepcopy(kwargs[param.arg])
        elif defaults[index] is not MISSING:
            local[param.arg] = deepcopy(defaults[index])
    bind_scope(func, local, root, eval_node, MISSING)
    return local


def _module_env(info: ModInfo, modules: dict[str, ModInfo], root: Path) -> dict[str, Any]:
    env: dict[str, Any] = {"__file__": str(info.path)}
    env.update(info.funcs)
    env.update({local: RefVal(canonical(target, modules)) for local, target in info.aliases.items()})
    env.update({name: RefVal(f"{info.name}.{name}") for name in info.classes})
    registries = reg_maps(info)
    env.update(registries)
    body = [item for item in info.tree.body if not isinstance(item, (ast.ClassDef, ast.FunctionDef))]
    bind_scope(ast.Module(body=body, type_ignores=[]), env, root, eval_node, MISSING)
    for name, values in registries.items():
        current = env.get(name)
        if isinstance(current, dict):
            current.update(values)
        else:
            env[name] = values
    return env
