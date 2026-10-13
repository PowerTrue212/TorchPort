"""与执行器无关的张量、包络、模块和指标比较。"""

from .common import SHAPE_PREFIX, _band, _diameter, _is_output, _l2, _leaf_key, _max_abs, _rel
from .envelope import (
    ENVELOPE_V3,
    compare_envelope,
    compare_envelope_v3,
    compare_noise_calibrated,
)
from .modules import (
    called_modules,
    compare_isolated_v3,
    compare_metric_envelope,
    compare_module_config,
)
from .tensors import TOLERANCE_PROFILES, compare_arrays, compare_tensor_archives, comparison_metrics

__all__ = [
    "ENVELOPE_V3",
    "SHAPE_PREFIX",
    "TOLERANCE_PROFILES",
    "_band",
    "_diameter",
    "_is_output",
    "_l2",
    "_leaf_key",
    "_max_abs",
    "_rel",
    "called_modules",
    "compare_arrays",
    "compare_envelope",
    "compare_envelope_v3",
    "compare_isolated_v3",
    "compare_metric_envelope",
    "compare_module_config",
    "compare_noise_calibrated",
    "compare_tensor_archives",
    "comparison_metrics",
]
