from __future__ import annotations

import argparse
import json
from typing import Any


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("形状必须是逗号分隔的整数") from exc
    if not shape or any(item <= 0 for item in shape):
        raise argparse.ArgumentTypeError("形状维度必须是正整数")
    return shape


def parse_list(value: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("必须是 JSON 数组") from exc
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("必须是 JSON 数组")
    return parsed


def parse_dict(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("必须是 JSON 对象") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("必须是 JSON 对象")
    return parsed
