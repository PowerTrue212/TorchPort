"""Harness 对 source/target 仓库的受控文件访问。"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from torchport.contracts import CODE_SUFFIXES

RESERVED_PARTS = {".git", ".torchport", "__pycache__", "node_modules"}
MAX_TEXT_BYTES = 1_000_000
MAX_READ_LINES = 200


class WorkspaceAccess:
    def __init__(
        self,
        target: Path,
        source: Callable[[], Path],
        candidate_identity: Callable[[], str],
    ) -> None:
        self.target = target.resolve()
        self._source = source
        self._candidate_identity = candidate_identity

    def root(self, side: str) -> Path:
        if side == "target":
            return self.target
        if side == "source":
            return self._source().resolve()
        raise ValueError("side 必须是 source 或 target")

    def path(self, value: str, *, side: str = "target", allow_directory: bool = False) -> Path:
        root = self.root(side)
        raw = Path(value)
        if raw.is_absolute():
            raise PermissionError("只允许仓库内的相对路径")
        path = (root / raw).resolve()
        if path != root and root not in path.parents:
            raise PermissionError(f"路径越过 {side} 工作区")
        relative = path.relative_to(root)
        if any(part in RESERVED_PARTS for part in relative.parts):
            raise PermissionError("该路径属于保留目录")
        if not allow_directory and path.exists() and path.is_dir():
            raise IsADirectoryError(value)
        return path

    def list_files(self, value: str, side: str) -> list[str]:
        side_root = self.root(side)
        root = self.path(value, side=side, allow_directory=True)
        if not root.is_dir():
            raise FileNotFoundError(value)
        files: list[str] = []
        for directory, names, names_files in os.walk(root, followlinks=False):
            names[:] = sorted(name for name in names if name not in RESERVED_PARTS)
            for name in sorted(names_files):
                path = Path(directory) / name
                if path.is_file():
                    files.append(path.relative_to(side_root).as_posix())
                    if len(files) >= 500:
                        return files
        return files

    def read_file(
        self,
        value: str,
        side: str,
        start_line: int | None,
        end_line: int | None,
    ) -> dict[str, Any]:
        root = self.root(side)
        path = self.path(value, side=side)
        if not path.is_file():
            raise FileNotFoundError(value)
        if path.stat().st_size > MAX_TEXT_BYTES:
            raise ValueError("文件过大")
        lines = path.read_text("utf-8").splitlines(keepends=True)
        total = len(lines)
        if total == 0:
            return {
                "side": side, "path": path.relative_to(root).as_posix(),
                "start_line": 1, "end_line": 0, "total_lines": 0,
                "truncated": False, "text": "",
            }
        start = 1 if start_line is None else start_line
        end = min(total, start + MAX_READ_LINES - 1) if end_line is None else end_line
        if start < 1 or end < start:
            raise ValueError("read_file 行号必须满足 1 <= start_line <= end_line")
        if end - start + 1 > MAX_READ_LINES:
            raise ValueError(f"read_file 每次最多读取 {MAX_READ_LINES} 行")
        return {
            "side": side,
            "path": path.relative_to(root).as_posix(),
            "start_line": start,
            "end_line": min(end, total),
            "total_lines": total,
            "truncated": end < total,
            "text": "".join(lines[start - 1 : end]),
        }

    def search_code(self, query: str, value: str, side: str) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query 不能为空")
        side_root = self.root(side)
        root = self.path(value, side=side, allow_directory=True)
        paths = [root] if root.is_file() else [
            Path(directory) / name
            for directory, _names, files in os.walk(root, followlinks=False)
            for name in files
            if not any(part in RESERVED_PARTS for part in Path(directory).relative_to(side_root).parts)
        ]
        matches: list[dict[str, Any]] = []
        for path in paths:
            if path.suffix.lower() not in CODE_SUFFIXES or path.stat().st_size > MAX_TEXT_BYTES:
                continue
            try:
                lines = path.read_text("utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, 1):
                if query in line:
                    matches.append({
                        "side": side, "path": path.relative_to(side_root).as_posix(),
                        "line": line_number, "text": line[:500],
                    })
                    if len(matches) >= 100:
                        return matches
        return matches

    def apply_patch(self, value: str, old: Any, new: str) -> dict[str, Any]:
        if not isinstance(new, str):
            raise TypeError("new 必须是字符串")
        path = self.path(value)
        if old is None:
            if path.exists():
                raise FileExistsError(value)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new, "utf-8")
            action = "created"
        else:
            if not isinstance(old, str) or not old:
                raise ValueError("old 必须是非空字符串，创建文件时使用 null")
            if not path.is_file():
                raise FileNotFoundError(value)
            text = path.read_text("utf-8")
            count = text.count(old)
            if count != 1:
                raise ValueError(f"old 必须精确出现一次，实际为 {count} 次")
            path.write_text(text.replace(old, new, 1), "utf-8")
            action = "updated"
        return {
            "path": path.relative_to(self.target).as_posix(),
            "action": action,
            "candidate_identity": self._candidate_identity(),
        }
