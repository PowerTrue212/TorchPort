from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    text: str
    rule: str


class CodeMap:
    def __init__(self, source: str) -> None:
        self.source = source
        self.lines = source.splitlines(keepends=True)
        self.starts: list[int] = []
        offset = 0
        for line in self.lines:
            self.starts.append(offset)
            offset += len(line)

    def offset(self, line: int, column: int) -> int:
        text = self.lines[line - 1]
        prefix = text.encode("utf-8")[:column].decode("utf-8")
        return self.starts[line - 1] + len(prefix)

    def span(self, node: ast.AST) -> tuple[int, int]:
        return (
            self.offset(node.lineno, node.col_offset),
            self.offset(node.end_lineno, node.end_col_offset),
        )

    def segment(self, node: ast.AST) -> str:
        start, end = self.span(node)
        return self.source[start:end]

    def attr_span(self, node: ast.Attribute) -> tuple[int, int]:
        end = self.offset(node.end_lineno, node.end_col_offset)
        return end - len(node.attr), end


def dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def finding(
    *,
    code: str,
    level: str,
    file: str,
    line: int | None,
    symbol: str | None,
    message: str,
    fix: str,
    automatic: bool = False,
) -> dict[str, Any]:
    return {
        "id": code,
        "severity": level,
        "file": file,
        "line": line,
        "symbol": symbol,
        "message": message,
        "suggestion": fix,
        "automatic": automatic,
    }
