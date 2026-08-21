from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import Any

Eval = Callable[[ast.AST, dict[str, Any], Path], Any]


def bind_scope(
    scope: ast.AST,
    env: dict[str, Any],
    root: Path,
    evaluate: Eval,
    missing: Any,
) -> None:
    for node in ast.walk(scope):
        if isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.optional_vars, ast.Name):
                    value = evaluate(item.context_expr, env, root)
                    if value is not missing:
                        env[item.optional_vars.id] = value
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value_node = node.value
            value = evaluate(value_node, env, root) if value_node else missing
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is not missing:
                for target in targets:
                    _assign(target, value, env, root, evaluate, missing)
        elif isinstance(node, ast.Expr):
            evaluate(node.value, env, root)


def run_func(
    func: ast.FunctionDef,
    args: list[Any],
    kwargs: dict[str, Any],
    env: dict[str, Any],
    root: Path,
    evaluate: Eval,
    missing: Any,
) -> Any:
    local = dict(env)
    params = func.args.args
    defaults = [missing] * (len(params) - len(func.args.defaults))
    defaults.extend(evaluate(item, env, root) for item in func.args.defaults)
    for index, param in enumerate(params):
        if index < len(args):
            local[param.arg] = args[index]
        elif param.arg in kwargs:
            local[param.arg] = kwargs[param.arg]
        elif defaults[index] is not missing:
            local[param.arg] = defaults[index]
        else:
            return missing
    bind_scope(func, local, root, evaluate, missing)
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            value = evaluate(node.value, local, root)
            if value is not missing:
                return value
    return missing


def scan_calls(
    tree: ast.Module,
    root: Path,
    path: Path,
    evaluate: Eval,
    missing: Any,
    initial: dict[str, Any] | None = None,
) -> list[tuple[ast.Call, dict[str, Any]]]:
    env: dict[str, Any] = {"__file__": str(path), **(initial or {})}
    env.update(
        {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
    )
    body = [item for item in tree.body if not isinstance(item, (ast.ClassDef, ast.FunctionDef))]
    module = ast.Module(body=body, type_ignores=[])
    bind_scope(module, env, root, evaluate, missing)
    result = []
    seen = set()
    scopes: list[ast.AST] = [module]
    scopes.extend(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    for scope in scopes:
        local = dict(env)
        bind_scope(scope, local, root, evaluate, missing)
        for node in ast.walk(scope):
            if isinstance(node, ast.Call) and id(node) not in seen:
                seen.add(id(node))
                result.append((node, local))
    return result


def _assign(
    target: ast.AST,
    value: Any,
    env: dict[str, Any],
    root: Path,
    evaluate: Eval,
    missing: Any,
) -> None:
    if isinstance(target, ast.Name):
        env[target.id] = value
        return
    if isinstance(target, ast.Attribute):
        owner = evaluate(target.value, env, root)
        if isinstance(owner, dict):
            owner[target.attr] = value
        return
    if isinstance(target, ast.Subscript):
        owner = evaluate(target.value, env, root)
        key = evaluate(target.slice, env, root)
        if owner is not missing and key is not missing:
            try:
                owner[key] = value
            except (KeyError, IndexError, TypeError):
                return
