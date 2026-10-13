"""Source 合法运行变体及批量结果归一化。"""

from __future__ import annotations

from typing import Any

import numpy as np

VARIANT_KEYS = {
    "name",
    "allow_tf32",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "dtype",
    "device",
    "channels_last",
    "batch",
    "batch_mode",
}
BATCH_MODES = ("copy", "distinct")


def apply_variant(model: Any, variant: dict[str, Any] | None) -> dict[str, Any] | None:
    """在 source 端施加一个合法运行变体，返回实际生效的后端设置。"""
    if not variant:
        return None
    unknown = set(variant) - VARIANT_KEYS
    if unknown:
        raise ValueError(f"未知的 source 变体字段：{', '.join(sorted(unknown))}")
    import torch

    backends = torch.backends
    if "allow_tf32" in variant:
        backends.cudnn.allow_tf32 = backends.cuda.matmul.allow_tf32 = bool(variant["allow_tf32"])
    if "cudnn_benchmark" in variant:
        backends.cudnn.benchmark = bool(variant["cudnn_benchmark"])
    if "cudnn_deterministic" in variant:
        backends.cudnn.deterministic = bool(variant["cudnn_deterministic"])
    dtype = variant.get("dtype")
    if dtype not in (None, "float32", "float64"):
        raise ValueError(f"source 变体 dtype 只允许 float32/float64：{dtype}")
    if dtype == "float64":
        model.double()
    device = variant.get("device")
    if device not in (None, "cpu", "cuda"):
        raise ValueError(f"source 变体 device 只允许 cpu/cuda：{device}")
    if device == "cpu":
        model.cpu()
    channels_last = bool(variant.get("channels_last", False))
    if channels_last:
        model.to(memory_format=torch.channels_last)
    batch = int(variant.get("batch", 1))
    if batch < 1:
        raise ValueError("source 变体 batch 必须为正整数")
    batch_mode = variant.get("batch_mode", "copy")
    if batch_mode not in BATCH_MODES:
        raise ValueError(f"source 变体 batch_mode 只允许 {'/'.join(BATCH_MODES)}：{batch_mode}")
    return {
        "name": variant.get("name"),
        "dtype": dtype or "float32",
        "device": device,
        "channels_last": channels_last,
        "batch": batch,
        "batch_mode": batch_mode,
        "cudnn_allow_tf32": backends.cudnn.allow_tf32,
        "matmul_allow_tf32": backends.cuda.matmul.allow_tf32,
        "cudnn_benchmark": backends.cudnn.benchmark,
        "cudnn_deterministic": backends.cudnn.deterministic,
    }


def _variant_inputs(tensors: list[Any], variant: dict[str, Any] | None) -> list[Any]:
    """按变体调整 source 输入：dtype、内存布局、批内复制。"""
    if not variant:
        return tensors
    import torch

    result = []
    for tensor in tensors:
        if variant["dtype"] == "float64" and tensor.is_floating_point():
            tensor = tensor.double()
        if variant["batch"] > 1:
            if variant.get("batch_mode", "copy") == "distinct" and tensor.dim() >= 2:
                # 第 0 份是原输入，其余份内容不同：批内统计或跨样本计算会改变第 0 份的结果
                others = [
                    torch.roll(torch.flip(tensor, dims=[-1]), shifts=i, dims=-1)
                    for i in range(variant["batch"] - 1)
                ]
                tensor = torch.cat([tensor, *others], dim=0)
            else:
                tensor = torch.cat([tensor] * variant["batch"], dim=0)
        if variant["channels_last"] and tensor.dim() == 4:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        result.append(tensor)
    return result


def _unbatch(values: dict[str, np.ndarray], batch: int, mode: str = "copy") -> dict[str, np.ndarray]:
    """批量变体只保留第一份；拆不开的张量丢弃（只少证据，不放宽包络）。

    copy：各份必须一致才保留。distinct：各份输入本就不同，只取第 0 份，
    批量无关性由 E1 v3 比较第 0 份与其他变体的距离来检验。
    """
    if batch == 1:
        return values
    kept: dict[str, np.ndarray] = {}
    for name, value in values.items():
        if value.ndim == 0 or value.shape[0] % batch:
            continue
        parts = np.split(value, batch, axis=0)
        if mode == "distinct" or all(
            np.allclose(parts[0], part, rtol=1e-4, atol=1e-6) for part in parts[1:]
        ):
            kept[name] = parts[0]
    return kept


_SKETCH_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}

# 模块构造超参：两端同名模块逐键比较，直接报出 eps / padding / align_corners 等事实性差异。

