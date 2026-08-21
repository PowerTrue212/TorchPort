from __future__ import annotations

from pathlib import Path


def import_paths(root: Path) -> list[Path]:
    paths = []
    source = root / "src"
    if source.is_dir():
        paths.append(source)
    paths.append(root)
    return paths


def module_name(path: Path, root: Path) -> str | None:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts.pop(0)
    if parts and parts[-1] == "__init__":
        parts.pop()
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    return ".".join(parts)
