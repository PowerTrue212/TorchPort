from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from .paths import module_name


def cli_hints(root: Path, entry: str | None) -> dict[str, Any]:
    if entry is None:
        return {}
    names = _names(root, entry)
    result: dict[str, Any] = {}
    for path in _files(root):
        for command in _commands(path.read_text(encoding="utf-8", errors="ignore")):
            if "-" not in command or not any(name in command for name in names):
                continue
            result.update(_flags(command, names))
    return result


def _names(root: Path, entry: str) -> set[str]:
    names = {Path(entry).name, entry}
    if (name := module_name(root / entry, root)) is not None:
        names.add(name)
    return names


def _files(root: Path) -> list[Path]:
    names = ("README.md", "README.rst", "README.txt", "readme.md", "readme.rst")
    return [path for name in names if (path := root / name).is_file()]


def _commands(text: str) -> list[str]:
    result = []
    parts = []
    for raw in text.splitlines():
        line = raw.strip().strip("`")
        if not line:
            continue
        if parts or line.endswith("\\"):
            parts.append(line.rstrip("\\").strip())
            if line.endswith("\\"):
                continue
            result.append(" ".join(parts))
            parts = []
        else:
            result.append(line)
    if parts:
        result.append(" ".join(parts))
    return result


def _flags(line: str, names: set[str]) -> dict[str, Any]:
    try:
        tokens = shlex.split(line.strip().strip("`"))
    except ValueError:
        return {}
    result: dict[str, Any] = {}
    index = _start(tokens, names)
    while index < len(tokens):
        token = tokens[index]
        parsed = _flag(token)
        if parsed is None:
            index += 1
            continue
        flag, value = parsed
        sep = value is not None
        if not sep:
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                value = tokens[index + 1]
                index += 1
            else:
                value = "true"
        result[flag.replace("-", "_")] = _value(value)
        index += 1
    return result


def _start(tokens: list[str], names: set[str]) -> int:
    for index, token in enumerate(tokens):
        if token in names:
            return index + 1
    return 0


def _flag(token: str) -> tuple[str, str | None] | None:
    if token.startswith("--"):
        flag, sep, value = token[2:].partition("=")
        return flag, value if sep else None
    if token.startswith("-") and len(token) > 1 and not token[1].isdigit():
        flag, sep, value = token[1:].partition("=")
        return flag, value if sep else None
    return None


def _value(text: str) -> Any:
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text
