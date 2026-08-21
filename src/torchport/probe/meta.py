from __future__ import annotations

import ast


def required(node: ast.FunctionDef, skip: int = 0) -> list[str]:
    args = node.args.args[skip:]
    count = len(args) - len(node.args.defaults)
    return [item.arg for item in args[:count]]


def init_func(model: ast.ClassDef) -> ast.FunctionDef | None:
    return next(
        (
            item
            for item in model.body
            if isinstance(item, ast.FunctionDef) and item.name == "__init__"
        ),
        None,
    )


def init_args(model: ast.ClassDef) -> list[str]:
    node = init_func(model)
    return required(node, 1) if node else []


def inputs(model: ast.ClassDef) -> list[str]:
    node = next(
        (
            item
            for item in model.body
            if isinstance(item, ast.FunctionDef) and item.name == "forward"
        ),
        None,
    )
    return [item.arg for item in node.args.args[1:]] if node else []
