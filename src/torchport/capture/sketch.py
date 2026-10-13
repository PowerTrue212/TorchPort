"""模块配置证据与层输出 CountSketch。"""

from __future__ import annotations

from typing import Any

import numpy as np

SKETCH_DIM = 1 << 16
_SKETCH_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


# 只记录标量或标量元组；张量属性（weight/bias 本身）不进入。
CONFIG_KEYS = (
    "eps", "momentum", "affine", "track_running_stats", "normalized_shape", "elementwise_affine",
    "num_groups", "num_features", "in_channels", "out_channels", "in_features", "out_features",
    "kernel_size", "stride", "padding", "dilation", "groups", "output_padding", "padding_mode",
    "output_size", "scale_factor", "mode", "align_corners", "ceil_mode", "count_include_pad",
    "approximate", "negative_slope", "num_heads",
)
_MISSING = object()


def _config_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (bool, int, float, str)) or item is None for item in value
    ):
        return list(value)
    return _MISSING


def module_config(model: Any) -> dict[str, dict[str, Any]]:
    """每个子模块的类名与构造超参；只取标量，缺失或非标量的属性不记录。

    类名只作报告（两端类名本来就不同），比较只看同名模块都有的超参键。
    """
    result: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        entry: dict[str, Any] = {"class": type(module).__name__}
        for key in CONFIG_KEYS:
            try:
                value = _config_value(getattr(module, key, _MISSING))
            except Exception:  # 个别模块的 __getattr__ 会抛非 AttributeError
                continue
            if value is not _MISSING:
                entry[key] = value
        result[name] = entry
    return result


def _is_output_name(name: str) -> bool:
    return name == "output" or name.startswith("output.")


def sketch_layers(
    values: dict[str, np.ndarray], keep: set[str] | frozenset[str] = frozenset()
) -> dict[str, np.ndarray]:
    """把每个张量压成固定维度的 CountSketch；原形状另存 `__shape__/<name>`。

    哈希只由元素数决定，双端同名同形状的张量落在同一投影上；投影线性，
    sketch(a) - sketch(b) = sketch(a - b)，差值范数无偏估计。小张量与 keep 中的张量原样保存。
    """
    result: dict[str, np.ndarray] = {}
    for name, value in values.items():
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        result[f"__shape__/{name}"] = np.asarray(value.shape, dtype=np.int64)
        if flat.size <= SKETCH_DIM or name in keep:
            result[name] = flat
            continue
        if flat.size not in _SKETCH_CACHE:
            rng = np.random.default_rng(flat.size)
            _SKETCH_CACHE[flat.size] = (
                rng.integers(0, SKETCH_DIM, flat.size),
                rng.choice(np.asarray([-1.0, 1.0]), flat.size),
            )
        bucket, sign = _SKETCH_CACHE[flat.size]
        result[name] = np.bincount(bucket, weights=sign * flat, minlength=SKETCH_DIM)
    return result
