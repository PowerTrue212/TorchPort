from __future__ import annotations

from collections.abc import Callable
from typing import Any


def flatten(
    value: Any,
    prefix: str,
    tensor: Callable[[Any], bool],
    convert: Callable[[Any], Any],
) -> dict[str, Any]:
    """按稳定的容器顺序展平张量叶子。"""
    if tensor(value):
        return {prefix: convert(value)}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    else:
        return {}
    result = {}
    for key, item in items:
        result.update(flatten(item, f"{prefix}.{key}", tensor, convert))
    return result
