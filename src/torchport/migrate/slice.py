from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .edit import CodeMap, Edit

BUILTINS = {"bool", "float", "int", "len", "max", "min", "print", "range", "str"}


def prune_imports(root: Path, probe: dict[str, Any]) -> dict[str, Any]:
    entry = probe.get("entry")
    model = probe.get("model")
    if not entry or not model:
        return {"rules": [], "files": [], "lines": {}, "names": [], "symbols": {}}
    path = root / entry
    if not path.is_file():
        return {"rules": [], "files": [], "lines": {}, "names": [], "symbols": {}}
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=entry)
    names = _closure(tree, _local(str(model)))
    rendered, lines, symbols, rules = _prune(text, tree, names)
    if rendered == text:
        return {"rules": [], "files": [], "lines": {}, "names": sorted(names), "symbols": {}}
    path.write_text(rendered, encoding="utf-8")
    return {
        "rules": sorted(rules),
        "files": [entry],
        "lines": {entry: sorted(lines)},
        "names": sorted(names),
        "symbols": {entry: sorted(symbols)},
    }


def _local(name: str) -> str:
    value = name.rsplit(":", maxsplit=1)[-1]
    return value.rsplit(".", maxsplit=1)[-1]


def _closure(tree: ast.Module, seed: str) -> set[str]:
    defs: dict[str, ast.AST] = {}
    imports = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in _stores(node):
                defs[name] = node
        elif isinstance(node, ast.Import | ast.ImportFrom):
            imports.update(_name(alias) for alias in node.names)

    used: set[str] = set()
    todo = [seed]
    while todo:
        name = todo.pop()
        if name in used:
            continue
        used.add(name)
        node = defs.get(name)
        if node is None:
            continue
        for ref in _loads(node):
            if ref in defs or ref in imports:
                todo.append(ref)
    return used


def _stores(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    result = []
    for target in targets:
        if isinstance(target, ast.Name):
            result.append(target.id)
    return result


def _loads(node: ast.AST) -> set[str]:
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }


def _name(alias: ast.alias) -> str:
    return alias.asname or alias.name.split(".", maxsplit=1)[0]


def _prune(text: str, tree: ast.Module, names: set[str]) -> tuple[str, set[int], set[str], set[str]]:
    code = CodeMap(text)
    edits: list[Edit] = []
    lines: set[int] = set()
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            keep = [alias for alias in node.names if _name(alias) in names]
            if len(keep) == len(node.names):
                continue
            lines.add(node.lineno)
            symbols.update(alias.name for alias in node.names if _name(alias) not in names)
            change = ast.unparse(ast.Import(names=keep)) if keep else ""
            start, end = code.span(node)
            edits.append(Edit(start, end, change, "scope.prune-import"))
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            keep = [alias for alias in node.names if _name(alias) in names]
            if len(keep) == len(node.names):
                continue
            lines.add(node.lineno)
            symbols.add(node.module or "")
            change = ast.unparse(ast.ImportFrom(module=node.module, names=keep, level=node.level))
            if not keep:
                change = ""
            start, end = code.span(node)
            edits.append(Edit(start, end, change, "scope.prune-import"))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in names:
                continue
            lines.update(range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1))
            symbols.add(node.name)
            start, end = code.span(node)
            edits.append(Edit(start, end, "", "scope.prune-code"))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            stores = set(_stores(node))
            if stores and stores <= names:
                continue
            lines.update(range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1))
            symbols.update(stores)
            start, end = code.span(node)
            edits.append(Edit(start, end, "", "scope.prune-code"))
        elif isinstance(node, ast.If):
            loaded = _loads(node)
            if _kept(loaded, names):
                continue
            lines.update(range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1))
            symbols.update(loaded)
            start, end = code.span(node)
            edits.append(Edit(start, end, "", "scope.prune-code"))
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            loaded = _loads(node)
            if _kept(loaded, names):
                continue
            lines.update(range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1))
            symbols.update(loaded)
            start, end = code.span(node)
            edits.append(Edit(start, end, "", "scope.prune-code"))
        elif not isinstance(node, ast.Pass):
            loaded = _loads(node)
            if _kept(loaded, names):
                continue
            lines.update(range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1))
            symbols.update(loaded)
            start, end = code.span(node)
            edits.append(Edit(start, end, "", "scope.prune-code"))
    rendered = text
    for edit in reversed(edits):
        rendered = rendered[: edit.start] + edit.text + rendered[edit.end :]
    return rendered, lines, symbols, {edit.rule for edit in edits}


def _kept(loaded: set[str], names: set[str]) -> bool:
    return bool(loaded) and (loaded - names - BUILTINS) == set()
