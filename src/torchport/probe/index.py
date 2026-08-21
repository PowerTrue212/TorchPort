from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .paths import module_name

SKIP_DIRS = {".git", ".venv", "venv", "env", "build", "dist", "artifacts", "tests"}


@dataclass(frozen=True)
class ModInfo:
    name: str
    path: Path
    tree: ast.Module
    classes: dict[str, ast.ClassDef]
    funcs: dict[str, ast.FunctionDef]
    aliases: dict[str, str]


def _models(tree: ast.Module) -> dict[str, ast.ClassDef]:
    result = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [ast.unparse(base) for base in node.bases]
        if any(base == "Module" or base.endswith(".Module") for base in bases):
            result[node.name] = node
    return result


def _package(name: str, path: Path) -> str:
    return name if path.name == "__init__.py" else name.rpartition(".")[0]


def _imports(tree: ast.Module, name: str, path: Path) -> dict[str, str]:
    result = {}
    package = _package(name, path)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                local = item.asname or item.name.split(".")[0]
                result[local] = item.name if item.asname else local
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                keep = max(len(parts) - node.level + 1, 0)
                base = ".".join([*parts[:keep], *base.split(".")]).strip(".")
            for item in node.names:
                if item.name == "*":
                    continue
                local = item.asname or item.name
                result[local] = ".".join(part for part in (base, item.name) if part)
    return result


def load_index(root: Path) -> dict[str, ModInfo]:
    result = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        name = module_name(path, root)
        if name is None:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        except (OSError, SyntaxError, UnicodeError):
            continue
        funcs = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        result[name] = ModInfo(
            name=name,
            path=path,
            tree=tree,
            classes=_models(tree),
            funcs=funcs,
            aliases=_imports(tree, name, path),
        )
    return result


def _split(symbol: str, modules: dict[str, ModInfo]) -> tuple[ModInfo, list[str]] | None:
    for name in sorted(modules, key=len, reverse=True):
        if symbol == name:
            return modules[name], []
        if symbol.startswith(f"{name}."):
            return modules[name], symbol[len(name) + 1 :].split(".")
    return None


def canonical(symbol: str, modules: dict[str, ModInfo]) -> str:
    for _step in range(8):
        split = _split(symbol, modules)
        if split is None:
            return symbol
        info, tail = split
        if not tail or tail[0] not in info.aliases:
            return symbol
        symbol = ".".join([info.aliases[tail[0]], *tail[1:]])
    return symbol


def expr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = expr_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def resolve(info: ModInfo, value: str, modules: dict[str, ModInfo]) -> str:
    first, dot, rest = value.partition(".")
    if first in info.aliases:
        symbol = info.aliases[first] + (f".{rest}" if dot else "")
    elif first in info.classes or first in info.funcs:
        symbol = f"{info.name}.{value}"
    else:
        symbol = value
    return canonical(symbol, modules)
