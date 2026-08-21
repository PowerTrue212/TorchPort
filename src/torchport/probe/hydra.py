from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .index import ModInfo, canonical, expr_name, resolve
from .refs import RefVal
from .values import call_args, eval_node

META = {"_target_", "_args_", "_convert_", "_recursive_", "_partial_"}
NAMES = {"hydra.utils.instantiate", "hydra.utils.call"}


def inst_ref(
    info: ModInfo,
    call: ast.Call,
    env: dict[str, Any],
    modules: dict[str, ModInfo],
    root: Path,
) -> tuple[str, tuple[list[Any], dict[str, Any]]] | None:
    if not _inst_name(info, call, env, modules, root):
        return None
    values = call_args(call, env, root)
    if values is None:
        return None
    args, kwargs = values
    if not args or not isinstance(args[0], dict):
        return None
    cfg = args[0]
    target = cfg.get("_target_")
    if not isinstance(target, str):
        return None
    inner_args = cfg.get("_args_", [])
    if not isinstance(inner_args, list):
        return None
    inner_kwargs = {key: value for key, value in cfg.items() if key not in META}
    inner_args.extend(args[1:])
    inner_kwargs.update({key: value for key, value in kwargs.items() if key not in META})
    return canonical(resolve(info, target, modules), modules), (inner_args, inner_kwargs)


def _inst_name(
    info: ModInfo,
    call: ast.Call,
    env: dict[str, Any],
    modules: dict[str, ModInfo],
    root: Path,
) -> bool:
    name = expr_name(call.func)
    target = resolve(info, name, modules) if name else None
    value = eval_node(call.func, env, root)
    if isinstance(value, RefVal):
        target = canonical(value.name, modules)
    return bool(target in NAMES or target == "instantiate")
