"""冻结输入的生成、加载与参数组装。"""

from __future__ import annotations

from typing import Any

import numpy as np


def input_key(index: int, spec: dict[str, Any]) -> str:
    arg, key = spec.get("arg"), spec.get("key")
    if spec.get("pack"):
        return f"input.{arg}.{spec['item']}" if arg else f"input.item.{spec['item']}"
    if key:
        return f"input.{arg}.{key}" if arg else f"input.{key}"
    if arg:
        return f"input.{arg}"
    return "input" if index == 0 else f"input.{index}"


def make_data(inputs: list[dict[str, Any]], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = {}
    for index, spec in enumerate(inputs):
        shape = tuple(spec["shape"])
        dtype = str(spec.get("dtype", "float32"))
        if spec.get("path"):
            value = np.load(spec["path"], allow_pickle=False)
            if not isinstance(value, np.ndarray):
                raise ValueError(f"固定输入不是数组：{spec['path']}")
            if tuple(value.shape) != shape or str(value.dtype) != dtype:
                raise ValueError(
                    f"固定输入 shape/dtype 不符：{spec['path']}，"
                    f"期望 {shape}/{dtype}，实际 {value.shape}/{value.dtype}"
                )
        elif dtype == "int64":
            value = rng.integers(0, 8, size=shape, dtype=np.int64)
        else:
            value = rng.standard_normal(shape).astype(np.float32)
        values[input_key(index, spec)] = value
    return values


def pack_values(values: list[Any], inputs: list[dict[str, Any]]) -> tuple[list, dict]:
    """按输入描述组装位置参数、关键字参数、字典和序列。"""
    sequences: dict[str | None, list[Any]] = {}
    kinds: dict[str | None, str] = {}
    for spec, value in zip(inputs, values):
        if spec.get("pack"):
            items = sequences.setdefault(spec.get("arg"), [])
            items.extend([None] * (spec["item"] + 1 - len(items)))
            items[spec["item"]] = value
            kinds[spec.get("arg")] = spec["pack"]
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    pos_dict: dict[str, Any] | None = None
    emitted: set[str | None] = set()
    for spec, value in zip(inputs, values):
        arg, key = spec.get("arg"), spec.get("key")
        if spec.get("pack"):
            if arg in emitted:
                continue
            emitted.add(arg)
            packed = sequences[arg]
            packed = tuple(packed) if kinds[arg] == "tuple" else packed
            (kwargs.__setitem__(arg, packed) if arg else args.append(packed))
        elif key and arg:
            kwargs.setdefault(arg, {})[key] = value
        elif key:
            if pos_dict is None:
                pos_dict = {}
                args.append(pos_dict)
            pos_dict[key] = value
        elif arg:
            kwargs[arg] = value
        else:
            args.append(value)
    return args, kwargs

