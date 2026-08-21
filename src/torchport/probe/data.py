from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def input_key(index: int, spec: dict[str, Any]) -> str:
    arg = spec.get("arg")
    key = spec.get("key")
    if spec.get("pack"):
        item = spec["item"]
        return f"input.{arg}.{item}" if arg else f"input.item.{item}"
    if key:
        return f"input.{arg}.{key}" if arg else f"input.{key}"
    if arg:
        return f"input.{arg}"
    return "input" if index == 0 else f"input.{index}"


def resolve_inputs(
    saved: dict[str, Any],
    inputs: list[dict[str, Any]] | None,
    shapes: tuple[tuple[int, ...], ...] | None,
    dtypes: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    if inputs:
        result = [dict(item) for item in inputs]
    elif shapes:
        types = dtypes or ("float32",)
        if len(types) == 1:
            types = types * len(shapes)
        if len(types) != len(shapes):
            raise ValueError("输入形状和类型数量不同")
        result = [
            {"shape": list(shape), "dtype": dtype}
            for shape, dtype in zip(shapes, types)
        ]
    elif saved.get("inputs"):
        result = [dict(item) for item in saved["inputs"]]
    elif saved.get("shape"):
        result = [{"shape": saved["shape"], "dtype": saved.get("dtype", "float32")}]
    else:
        return []
    for spec in result:
        shape = spec.get("shape")
        if not isinstance(shape, list) or not shape or any(
            not isinstance(value, int) or value <= 0 for value in shape
        ):
            raise ValueError("输入 shape 必须是正整数数组")
        if spec.get("dtype", "float32") not in {"float32", "int64"}:
            raise ValueError("输入 dtype 只支持 float32 或 int64")
        spec.setdefault("dtype", "float32")
        pack = spec.get("pack")
        item = spec.get("item")
        if (pack is None) != (item is None):
            raise ValueError("序列输入必须同时提供 pack 和 item")
        if pack not in {None, "list", "tuple"}:
            raise ValueError("输入 pack 只支持 list 或 tuple")
        if item is not None and (not isinstance(item, int) or item < 0):
            raise ValueError("输入 item 必须是非负整数")
        if pack and spec.get("key") is not None:
            raise ValueError("序列输入不能同时使用 key")
    keys = [input_key(index, item) for index, item in enumerate(result)]
    if len(keys) != len(set(keys)):
        raise ValueError("输入名称或字典键重复")
    groups: dict[str | None, list[dict[str, Any]]] = {}
    modes: dict[str, str] = {}
    for spec in result:
        if spec.get("pack"):
            groups.setdefault(spec.get("arg"), []).append(spec)
        arg = spec.get("arg")
        if arg:
            mode = spec.get("pack") or ("dict" if spec.get("key") else "value")
            previous = modes.setdefault(arg, mode)
            if previous != mode:
                raise ValueError(f"关键字输入 {arg} 使用了冲突的容器类型")
    for group in groups.values():
        packs = {item["pack"] for item in group}
        slots = sorted(item["item"] for item in group)
        if len(packs) != 1 or slots != list(range(len(slots))):
            raise ValueError("同一序列必须类型一致且 item 连续")
    return result


def make_data(inputs: list[dict[str, Any]], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = {}
    for index, spec in enumerate(inputs):
        shape = tuple(spec["shape"])
        if spec["dtype"] == "int64":
            value = rng.integers(0, 8, size=shape, dtype=np.int64)
        else:
            value = rng.standard_normal(shape).astype(np.float32)
        values[input_key(index, spec)] = value
    return values


def load_data(path: Path, inputs: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            input_key(index, spec): np.asarray(archive[input_key(index, spec)])
            for index, spec in enumerate(inputs)
        }


def to_values(
    values: dict[str, np.ndarray],
    inputs: list[dict[str, Any]],
    kind: str,
) -> tuple[list[Any], dict[str, Any]]:
    if kind == "source":
        import torch

        convert = torch.from_numpy
    else:
        import jittor as jt

        convert = jt.array

    converted = [convert(values[input_key(index, spec)]) for index, spec in enumerate(inputs)]
    return pack_values(converted, inputs)


def pack_values(
    values: list[Any],
    inputs: list[dict[str, Any]],
) -> tuple[list[Any], dict[str, Any]]:
    """按输入描述组装位置参数、关键字参数、字典和序列。"""
    sequences: dict[str | None, list[Any]] = {}
    kinds: dict[str | None, str] = {}
    for spec, value in zip(inputs, values):
        pack = spec.get("pack")
        if not pack:
            continue
        token = spec.get("arg")
        items = sequences.setdefault(token, [None] * (spec["item"] + 1))
        if len(items) <= spec["item"]:
            items.extend([None] * (spec["item"] + 1 - len(items)))
        items[spec["item"]] = value
        kinds[token] = pack

    packed = {
        token: tuple(items) if kinds[token] == "tuple" else items
        for token, items in sequences.items()
    }
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    pos_dict: dict[str, Any] | None = None
    emitted: set[str | None] = set()
    for spec, value in zip(inputs, values):
        arg = spec.get("arg")
        key = spec.get("key")
        if spec.get("pack"):
            if arg in emitted:
                continue
            emitted.add(arg)
            if arg:
                kwargs[arg] = packed[arg]
            else:
                args.append(packed[arg])
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
