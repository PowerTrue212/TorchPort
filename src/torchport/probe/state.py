from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

STATE_KEYS = ("state_dict", "model_state_dict", "model", "net", "weights")
PREFIXES = ("module.", "_orig_mod.", "model.")
SKIP_KEYS = ("num_batches_tracked",)


def _portable(name: str) -> bool:
    return not name.endswith(SKIP_KEYS)


def _state(value: Any) -> dict[str, Any]:
    if hasattr(value, "state_dict"):
        value = value.state_dict()
    if not isinstance(value, dict):
        raise ValueError("检查点不是状态字典")
    for key in STATE_KEYS:
        item = value.get(key)
        if hasattr(item, "state_dict"):
            item = item.state_dict()
        if isinstance(item, dict):
            value = item
            break
    return {str(name): item for name, item in value.items() if hasattr(item, "shape")}


def _rename(weights: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        name[len(prefix) :] if name.startswith(prefix) else name: value
        for name, value in weights.items()
    }


def load_checkpoint(model: Any, path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise ValueError(f"源检查点不存在：{path}")
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ValueError("读取 safetensors 需要安装 safetensors") from exc
        raw = load_file(str(path), device="cpu")
    else:
        try:
            raw = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            raw = torch.load(path, map_location="cpu")
    weights = _state(raw)
    target = model.state_dict()
    choices = [weights, *(_rename(weights, prefix) for prefix in PREFIXES)]

    def matched(items: dict[str, Any]) -> dict[str, Any]:
        return {
            name: value
            for name, value in items.items()
            if name in target and tuple(value.shape) == tuple(target[name].shape)
        }

    selected = max(choices, key=lambda items: len(matched(items)))
    loaded = matched(selected)
    if not loaded:
        raise ValueError(f"检查点没有可加载的模型权重：{path}")
    model.load_state_dict(loaded, strict=False)
    missing = sorted(name for name in set(target) - set(loaded) if _portable(name))
    unexpected = sorted(name for name in set(selected) - set(loaded) if _portable(name))
    return {
        "requested": True,
        "path": str(path),
        "tensors": sum(_portable(name) for name in selected),
        "loaded": sum(_portable(name) for name in loaded),
        "total": sum(_portable(name) for name in target),
        "missing": missing,
        "unexpected": unexpected,
        "complete": not missing,
    }


def save_state(model, path: Path) -> int:
    weights = {
        name: item.detach().cpu().numpy()
        for name, item in model.state_dict().items()
        if _portable(name)
    }
    np.savez(path, **weights)
    return len(weights)


def load_state(model, path: Path) -> tuple[int, int]:
    with np.load(path, allow_pickle=False) as archive:
        weights = {
            name: np.asarray(archive[name])
            for name in archive.files
            if _portable(name)
        }
    target = model.state_dict()
    matched = {
        name: value
        for name, value in weights.items()
        if name in target and tuple(value.shape) == tuple(target[name].shape)
    }
    model.load_parameters(matched)
    return len(weights), len(matched)
