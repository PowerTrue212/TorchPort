from __future__ import annotations

import ast
from pathlib import Path

from torchport.probe.paths import module_name


def model_scope(root: Path, entry: str | None) -> set[str]:
    if entry is None:
        return set()
    modules = {
        name: path
        for path in root.rglob("*.py")
        if (name := module_name(path, root)) is not None
    }
    start = module_name(root / entry, root)
    if start is None or start not in modules:
        return {entry}
    result: set[str] = set()
    seen: set[str] = set()
    _walk(root, start, modules, seen, result)
    return result


def _walk(
    root: Path,
    name: str,
    modules: dict[str, Path],
    seen: set[str],
    result: set[str],
) -> None:
    if name in seen or name not in modules:
        return
    seen.add(name)
    path = modules[name]
    result.add(path.relative_to(root).as_posix())
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
    except (OSError, SyntaxError, UnicodeError):
        return
    for target in _imports(name, tree, modules):
        _walk(root, target, modules, seen, result)


def _imports(
    name: str,
    tree: ast.Module,
    modules: dict[str, Path],
) -> set[str]:
    result = set()
    package = name.rpartition(".")[0]
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name in modules:
                    result.add(item.name)
        elif isinstance(node, ast.ImportFrom):
            base = _base(package, node)
            if base in modules:
                result.add(base)
            for item in node.names:
                target = ".".join(part for part in (base, item.name) if part)
                if target in modules:
                    result.add(target)
    return result


def _base(package: str, node: ast.ImportFrom) -> str:
    base = node.module or ""
    if not node.level:
        return base
    parts = package.split(".") if package else []
    keep = max(len(parts) - node.level + 1, 0)
    return ".".join([*parts[:keep], *base.split(".")]).strip(".")
