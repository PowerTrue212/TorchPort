from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .envvals import EnvVals
from .factory import external, factory, model_args, return_args
from .hints import param_defaults, param_hints
from .hydra import inst_ref
from .index import ModInfo, canonical, expr_name, load_index, resolve
from .meta import init_args, init_func, inputs, required
from .modenv import import_funcs
from .paths import module_name
from .readme import cli_hints
from .refs import RefVal
from .scopes import scan_calls
from .shapes import input_specs
from .values import MISSING, call_args, eval_node


@dataclass(frozen=True)
class ModelRef:
    entry: str
    name: str
    model: ast.ClassDef
    tree: ast.Module
    args: list[Any]
    kwargs: dict[str, Any]
    model_args: list[Any]
    model_kwargs: dict[str, Any]
    required: list[str]
    env: dict[str, str | None]


def _initial(
    info: ModInfo,
    root: Path,
    entry: str,
    env_state: EnvVals,
    modules: dict[str, ModInfo],
    funcs: dict[str, tuple[ModInfo, ast.FunctionDef]],
) -> dict[str, Any]:
    refs: dict[str, Any] = {
        "__env__": env_state,
        "__cli__": cli_hints(root, entry),
    }
    refs.update({local: RefVal(target) for local, target in info.aliases.items()})
    refs.update({name: RefVal(f"{info.name}.{name}") for name in info.classes})
    refs.update(import_funcs(info, modules, funcs, root))
    return refs


def _target(
    info: ModInfo,
    call: ast.Call,
    env: dict[str, Any],
    modules: dict[str, ModInfo],
    root: Path,
    known: set[str],
) -> tuple[str | None, bool]:
    name = expr_name(call.func)
    target = resolve(info, name, modules) if name else None
    if target in known:
        return target, False
    value = eval_node(call.func, env, root)
    if isinstance(value, RefVal):
        return canonical(value.name, modules), True
    return target, False


def _config_values(values: tuple[list[Any], dict[str, Any]] | None) -> bool:
    return bool(values and values[0] and isinstance(values[0][0], dict))


def _reference(
    info: ModInfo,
    name: str,
    model: ast.ClassDef,
    call: ast.Call | None,
    env: dict[str, Any] | None,
    required: list[str],
    root: Path,
    env_state: EnvVals,
    *,
    load_info: ModInfo | None = None,
    load_name: str | None = None,
    call_values: tuple[list[Any], dict[str, Any]] | None = None,
    model_values: tuple[list[Any], dict[str, Any]] | None = None,
) -> ModelRef:
    values = call_values if call_values is not None else (
        call_args(call, env or {}, root) if call else None
    )
    args, kwargs = values or ([], {})
    model_args, model_kwargs = model_values or (args, kwargs)
    owner = load_info or info
    return ModelRef(
        entry=owner.path.relative_to(root).as_posix(),
        name=load_name or name,
        model=model,
        tree=owner.tree,
        args=args,
        kwargs=kwargs,
        model_args=model_args,
        model_kwargs=model_kwargs,
        required=[] if call is not None and values is not None else required,
        env=dict(env_state.used),
    )


def find_model(
    root: Path,
    entry: str,
    env: dict[str, str] | None = None,
) -> ModelRef | None:
    modules = load_index(root)
    entry_name = module_name(root / entry, root)
    info = modules.get(entry_name or "")
    if info is None:
        return None
    classes = {
        f"{module.name}.{name}": (module, name, node)
        for module in modules.values()
        for name, node in module.classes.items()
    }
    funcs = {
        f"{module.name}.{name}": (module, node)
        for module in modules.values()
        for name, node in module.funcs.items()
    }
    env_state = EnvVals(dict(env or {}))
    calls = scan_calls(
        info.tree,
        root,
        info.path,
        eval_node,
        MISSING,
        _initial(info, root, entry, env_state, modules, funcs),
    )
    known = set(classes) | set(funcs)

    for call, env in calls:
        inst = inst_ref(info, call, env, modules, root)
        if inst is None:
            continue
        target, values = inst
        if target in funcs:
            owner, func = funcs[target]
            model = factory(owner, func, modules, classes, root, values, env)
            if model is not None:
                return _reference(
                    info,
                    func.name,
                    model,
                    call,
                    env,
                    required(func),
                    root,
                    env_state,
                    load_info=owner if owner is not info else None,
                    load_name=func.name if owner is not info else None,
                    call_values=values,
                    model_values=return_args(owner, func, values, env, root, modules),
                )
        if target in classes:
            owner, class_name, model = classes[target]
            return _reference(
                info,
                class_name,
                model,
                call,
                env,
                init_args(model),
                root,
                env_state,
                load_info=owner if owner is not info else None,
                load_name=class_name if owner is not info else None,
                call_values=values,
            )

    for call, env in calls:
        target, dynamic = _target(info, call, env, modules, root, known)
        if target not in funcs:
            continue
        owner, func = funcs[target]
        values = call_args(call, env, root)
        model = factory(owner, func, modules, classes, root, values, env)
        if model is not None:
            name = expr_name(call.func) or func.name
            use_owner = dynamic or external(target, owner, info) or _config_values(values)
            return _reference(
                info,
                name,
                model,
                call,
                env,
                required(func),
                root,
                env_state,
                load_info=owner if use_owner and owner is not info else None,
                load_name=func.name if use_owner and owner is not info else None,
                call_values=values,
                model_values=model_args(owner, func, call, env, root, modules),
            )

    for call, env in calls:
        target, _dynamic = _target(info, call, env, modules, root, known)
        if target in classes:
            owner, class_name, model = classes[target]
            name = expr_name(call.func) or class_name
            return _reference(
                info,
                name,
                model,
                call,
                env,
                init_args(model),
                root,
                env_state,
                load_info=owner if owner is not info else None,
                load_name=class_name if owner is not info else None,
            )

    for local, target in info.aliases.items():
        target = canonical(target, modules)
        if target in classes:
            owner, class_name, model = classes[target]
            return _reference(
                info,
                local,
                model,
                None,
                None,
                init_args(model),
                root,
                env_state,
                load_info=owner if owner is not info else None,
                load_name=class_name if owner is not info else None,
            )
        if target in funcs:
            owner, func = funcs[target]
            if (model := factory(owner, func, modules, classes, root)) is not None:
                return _reference(
                    info, local, model, None, None, required(func), root, env_state
                )

    for module in modules.values():
        for name, model in module.classes.items():
            if not init_args(model):
                return _reference(
                    module, name, model, None, None, [], root, env_state
                )
    return None


def model_choices(
    root: Path,
    entry: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    modules = load_index(root)
    entry_name = module_name(root / entry, root) if entry else None
    classes = {
        f"{module.name}.{name}": (module, name, node)
        for module in modules.values()
        for name, node in module.classes.items()
    }
    model_names = {id(node): name for name, (_module, _name, node) in classes.items()}
    choices: list[dict[str, Any]] = []
    for module in modules.values():
        for name, func in module.funcs.items():
            model = factory(module, func, modules, classes, root)
            if model is None:
                continue
            choices.append(
                _choice(
                    root,
                    module,
                    func,
                    model,
                    f"{module.name}.{name}",
                    "factory",
                    required(func),
                    inputs(model),
                    entry_name,
                    returns=model_names.get(id(model)),
                )
            )
        for name, model in module.classes.items():
            init = init_func(model)
            choices.append(
                _choice(
                    root,
                    module,
                    init,
                    model,
                    f"{module.name}.{name}",
                    "class",
                    init_args(model),
                    inputs(model),
                    entry_name,
                )
            )
    choices.sort(key=lambda item: (-item["score"], item["name"]))
    result = [{key: value for key, value in item.items() if key != "score"} for item in choices]
    return result[:limit]


def _choice(
    root: Path,
    module: ModInfo,
    func: ast.FunctionDef | None,
    model: ast.ClassDef,
    name: str,
    kind: str,
    required: list[str],
    inputs: list[str],
    entry: str | None,
    *,
    returns: str | None = None,
) -> dict[str, Any]:
    score = 2 if module.name == entry else 0
    score += 1 if not required else 0
    result: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "file": module.path.relative_to(root).as_posix(),
        "required": required,
        "inputs": inputs,
        "score": score,
    }
    skip = 1 if kind == "class" else 0
    defaults = param_defaults(func, root, module, skip)
    hints = param_hints(root, module, required)
    example = {key: hints[key] for key in required if key in hints}
    specs = input_specs(model, {**defaults, **hints})
    if defaults:
        result["defaults"] = defaults
    if hints:
        result["hints"] = hints
    if example:
        result["example_kw"] = example
    if specs:
        result["input_specs"] = specs
    if returns:
        result["returns"] = returns
    return result
