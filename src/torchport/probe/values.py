from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from .argvals import ArgVals
from .envvals import EnvVals
from .refs import CfgVal, DictVal, FuncVal, RefVal
from .scopes import run_func

MISSING = object()


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _path(value: Any, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def eval_node(node: ast.AST, env: dict[str, Any], root: Path) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        pass
    if isinstance(node, ast.Name):
        if node.id in {"int", "float", "str", "bool"}:
            return {"int": int, "float": float, "str": str, "bool": bool}[node.id]
        if node.id == "Path":
            return Path
        return env.get(node.id, MISSING)
    if isinstance(node, ast.Attribute) and _name(node) == "pathlib.Path":
        return Path
    if isinstance(node, ast.List):
        values = [eval_node(item, env, root) for item in node.elts]
        return MISSING if MISSING in values else values
    if isinstance(node, ast.Tuple):
        values = [eval_node(item, env, root) for item in node.elts]
        return MISSING if MISSING in values else tuple(values)
    if isinstance(node, ast.Dict):
        result = {}
        for key_node, value_node in zip(node.keys, node.values):
            value = eval_node(value_node, env, root)
            if value is MISSING:
                return MISSING
            if key_node is None:
                if not isinstance(value, dict):
                    return MISSING
                result.update(value)
                continue
            key = eval_node(key_node, env, root)
            if key is MISSING:
                return MISSING
            result[key] = value
        return result
    if isinstance(node, ast.Attribute):
        if _name(node) == "os.environ":
            return env.get("__env__", MISSING)
        value = eval_node(node.value, env, root)
        if isinstance(value, RefVal):
            if node.attr == "__dict__":
                return DictVal(value.name)
            return RefVal(f"{value.name}.{node.attr}")
        if isinstance(value, dict):
            return value.get(node.attr, MISSING)
        if isinstance(value, Path) and node.attr == "parent":
            return value.parent
        return MISSING
    if isinstance(node, ast.Subscript):
        value = eval_node(node.value, env, root)
        key = eval_node(node.slice, env, root)
        if isinstance(value, EnvVals):
            return value.read(key, MISSING, MISSING)
        if isinstance(value, DictVal) and isinstance(key, str):
            return RefVal(f"{value.name}.{key}")
        try:
            return value[key]
        except (KeyError, IndexError, TypeError):
            return MISSING
    if isinstance(node, ast.Call):
        return _call(node, env, root)
    return MISSING


def _call(node: ast.Call, env: dict[str, Any], root: Path) -> Any:
    name = _name(node.func)
    args = [eval_node(item, env, root) for item in node.args]
    if MISSING in args:
        return MISSING
    kwargs = {
        item.arg: eval_node(item.value, env, root)
        for item in node.keywords
        if item.arg is not None
    }
    if MISSING in kwargs.values():
        return MISSING
    static = env.get(name or "")
    if _import_call(name, static) and args and isinstance(args[0], str):
        return RefVal(args[0])
    if isinstance(static, FuncVal):
        local = dict(static.env)
        local.update({key: value for key, value in env.items() if key.startswith("__")})
        return run_func(static.func, args, kwargs, local, root, eval_node, MISSING)
    if isinstance(static, ast.FunctionDef):
        return run_func(static, args, kwargs, env, root, eval_node, MISSING)
    if _cfg_node(name, static):
        value = args[0] if args and isinstance(args[0], dict) else {}
        return CfgVal(value)
    if name in {"int", "float", "str", "bool"}:
        convert = {"int": int, "float": float, "str": str, "bool": bool}[name]
        if not args:
            return convert
        try:
            return convert(args[0])
        except (TypeError, ValueError):
            return MISSING
    if name in {"argparse.ArgumentParser", "ArgumentParser"}:
        return ArgVals(hints=env.get("__cli__", {}))
    if name == "getattr" and len(args) >= 2:
        owner, attr = args[:2]
        if isinstance(owner, RefVal) and isinstance(attr, str):
            return RefVal(f"{owner.name}.{attr}")
    if name in {"os.getenv", "os.environ.get"} and args:
        state = env.get("__env__")
        if isinstance(state, EnvVals):
            default = args[1] if len(args) > 1 else MISSING
            return state.read(args[0], default, MISSING)
    if name in {"Path", "pathlib.Path"} and args:
        return _path(args[0], root)
    if name == "open" and args:
        return _path(args[0], root)
    if name == "json.loads" and args:
        try:
            return json.loads(args[0])
        except (TypeError, ValueError):
            return MISSING
    if name in {"OmegaConf.load", "omegaconf.OmegaConf.load"} and args:
        value = args[0]
        if isinstance(value, (str, Path)):
            try:
                value = _path(value, root).read_text(encoding="utf-8")
            except OSError:
                return MISSING
        return _yaml(value)
    if name in {
        "OmegaConf.to_container",
        "OmegaConf.to_object",
        "omegaconf.OmegaConf.to_container",
        "omegaconf.OmegaConf.to_object",
    } and args:
        return args[0]
    if name in {"json.load", "yaml.safe_load", "yaml.load"} and args:
        value = args[0]
        if isinstance(value, Path):
            try:
                value = value.read_text(encoding="utf-8")
            except OSError:
                return MISSING
        if name == "json.load":
            try:
                return json.loads(value)
            except (TypeError, ValueError):
                return MISSING
        return _yaml(value)
    if name in {"dict", "list", "tuple"}:
        try:
            base = args[0] if args else ({} if name == "dict" else [])
            value = {**base, **kwargs} if name == "dict" else base
            return {"dict": dict, "list": list, "tuple": tuple}[name](value)
        except (TypeError, ValueError):
            return MISSING
    if isinstance(node.func, ast.Attribute):
        owner = eval_node(node.func.value, env, root)
        method = node.func.attr
        try:
            if isinstance(owner, CfgVal) and method in {"clone", "copy"}:
                return owner.clone()
            if isinstance(owner, CfgVal) and method == "merge_from_file" and args:
                owner.merge_file(_path(args[0], root))
                return None
            if isinstance(owner, CfgVal) and method == "merge_from_list" and args:
                owner.merge_list(args[0])
                return None
            if isinstance(owner, ArgVals) and method == "add_argument":
                owner.add(args, kwargs)
                return None
            if isinstance(owner, ArgVals) and method == "set_defaults":
                owner.set(kwargs)
                return None
            if isinstance(owner, ArgVals) and method in {"parse_args", "parse_known_args"}:
                return owner.parse(method == "parse_known_args")
            if isinstance(owner, Path) and method in {"with_name", "joinpath"}:
                return getattr(owner, method)(*args)
            if isinstance(owner, Path) and method == "read_text":
                return owner.read_text(encoding=kwargs.get("encoding", "utf-8"))
            if isinstance(owner, dict) and method == "copy":
                return dict(owner)
            if isinstance(owner, dict) and method == "get":
                return owner.get(*args)
            if isinstance(owner, dict) and method == "pop":
                return owner.pop(*args)
        except (OSError, TypeError, ValueError):
            return MISSING
    return MISSING


def _cfg_node(name: str | None, static: Any) -> bool:
    targets = {"yacs.config.CfgNode"}
    return name in targets or (
        isinstance(static, RefVal) and static.name in targets
    )


def _import_call(name: str | None, static: Any) -> bool:
    return name in {"importlib.import_module", "import_module"} or (
        isinstance(static, RefVal) and static.name == "importlib.import_module"
    )


def _yaml(text: str) -> Any:
    try:
        import yaml
    except ImportError:
        return MISSING
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return MISSING


def call_args(
    call: ast.Call,
    env: dict[str, Any],
    root: Path,
) -> tuple[list[Any], dict[str, Any]] | None:
    args = []
    for node in call.args:
        if isinstance(node, ast.Starred):
            value = eval_node(node.value, env, root)
            if not isinstance(value, (list, tuple)):
                return None
            args.extend(value)
        else:
            value = eval_node(node, env, root)
            if value is MISSING:
                return None
            args.append(value)
    kwargs = {}
    for item in call.keywords:
        value = eval_node(item.value, env, root)
        if value is MISSING:
            return None
        if item.arg is None:
            if not isinstance(value, dict):
                return None
            kwargs.update(value)
        else:
            kwargs[item.arg] = value
    try:
        json.dumps({"args": args, "kwargs": kwargs})
    except (TypeError, ValueError):
        return None
    return args, kwargs
