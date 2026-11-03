"""固定容差下的张量归档与数组比较。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import SHAPE_PREFIX

TOLERANCE_PROFILES: dict[str, tuple[float, float]] = {
    "operator-strict": (1e-5, 1e-6),
    "fp32-inference": (1e-4, 1e-4),
}
def compare_tensor_archives(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    profile: str = "fp32-inference",
    rtol: float | None = None,
    atol: float | None = None,
    require_all: bool = True,
    skip_sketched: bool = False,
) -> dict[str, Any]:
    """比较两个 NPZ 证据包，并按参考包顺序定位首次分歧。"""
    reference_path = Path(reference_path).expanduser().resolve()
    candidate_path = Path(candidate_path).expanduser().resolve()
    if profile not in TOLERANCE_PROFILES:
        raise ValueError(f"未知容差配置：{profile}")
    profile_rtol, profile_atol = TOLERANCE_PROFILES[profile]
    rtol = profile_rtol if rtol is None else rtol
    atol = profile_atol if atol is None else atol
    if rtol < 0 or atol < 0:
        raise ValueError("rtol 和 atol 必须是非负数。")
    if not reference_path.is_file():
        raise ValueError(f"参考证据不存在：{reference_path}")
    if not candidate_path.is_file():
        raise ValueError(f"候选证据不存在：{candidate_path}")

    comparisons: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    first_gating_divergence: dict[str, Any] | None = None
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        candidate_path,
        allow_pickle=False,
    ) as candidate:
        reference_names = [name for name in reference.files if not name.startswith(SHAPE_PREFIX)]
        candidate_names = [name for name in candidate.files if not name.startswith(SHAPE_PREFIX)]
        if profile == "fp32-inference" and not any(
            name == "output" or name.startswith("output.") for name in reference_names
        ):
            raise ValueError("fp32-inference 参考证据必须包含 output 或 output.*。")
        missing = [name for name in reference_names if name not in candidate]
        extra = [name for name in candidate_names if name not in reference]

        for position, name in enumerate(reference_names):
            if not require_all and name not in candidate:
                continue
            gating = (
                profile == "operator-strict"
                or name == "output"
                or name.startswith(("output.", "input."))
                or name == "input"
            )
            if name not in candidate:
                item = {
                    "name": name,
                    "position": position,
                    "passed": False,
                    "reason": "候选证据缺少该张量",
                    "reference_elements": int(reference[name].size),
                }
            else:
                ref_value, cand_value = reference[name], candidate[name]
                shape_key = SHAPE_PREFIX + name
                ref_has_shape = shape_key in reference
                cand_has_shape = shape_key in candidate
                ref_shape = (
                    tuple(int(x) for x in reference[shape_key])
                    if ref_has_shape
                    else ref_value.shape
                )
                cand_shape = (
                    tuple(int(x) for x in candidate[shape_key])
                    if cand_has_shape
                    else cand_value.shape
                )
                sketched = ref_has_shape and int(np.prod(ref_shape)) > int(ref_value.size)
                if ref_has_shape != cand_has_shape:
                    item = {
                        "passed": False,
                        "reason": "一端缺少 CountSketch 原始形状",
                        "reference_shape": list(ref_shape),
                        "candidate_shape": list(cand_shape),
                    }
                elif ref_has_shape and ref_shape != cand_shape:
                    item = {
                        "passed": False,
                        "reason": "CountSketch 前的张量形状不同",
                        "reference_shape": list(ref_shape),
                        "candidate_shape": list(cand_shape),
                    }
                elif ref_has_shape and ref_value.shape != cand_value.shape:
                    item = {
                        "passed": False,
                        "reason": "CountSketch 存储形状不同",
                        "reference_shape": list(ref_value.shape),
                        "candidate_shape": list(cand_value.shape),
                    }
                elif sketched and skip_sketched:
                    item = {
                        "passed": True,
                        "skipped": True,
                        "storage": "countsketch",
                        "reason": "CountSketch 不参与逐元素严格比较",
                        "reference_shape": list(ref_shape),
                        "candidate_shape": list(cand_shape),
                        "stored_elements": int(ref_value.size),
                    }
                else:
                    item = compare_arrays(
                        ref_value,
                        cand_value,
                        rtol=rtol,
                        atol=atol,
                    )
                    item["storage"] = "countsketch" if sketched else "exact"
                    item["original_shape"] = list(ref_shape)
                item.update({"name": name, "position": position})
            item["gating"] = gating and not item.get("skipped", False)
            comparisons.append(item)
            if first_divergence is None and not item["passed"]:
                first_divergence = {
                    "name": name,
                    "position": position,
                    "reason": item.get("reason", "数值超出容差"),
                }
            if first_gating_divergence is None and gating and not item["passed"]:
                first_gating_divergence = {
                    "name": name,
                    "position": position,
                    "reason": item.get("reason", "数值超出容差"),
                }

    report = {
        "schema_version": 1,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "rtol": rtol,
        "atol": atol,
        "profile": profile,
        "passed": first_gating_divergence is None and (require_all or bool(comparisons)),
        "require_all": require_all,
        "skipped_sketched": [item["name"] for item in comparisons if item.get("skipped")],
        "first_divergence": first_divergence,
        "first_gating_divergence": first_gating_divergence,
        "missing_in_candidate": missing,
        "extra_in_candidate": extra,
        "tensors": comparisons,
    }
    report["metrics"] = comparison_metrics(report)
    return report


def comparison_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """把逐张量证据归约成稳定、可聚合的计数；结构不一致不伪造元素误差。"""
    tensors = list(report.get("tensors") or [])
    skipped = [item for item in tensors if item.get("skipped")]
    measured = [item for item in tensors if not item.get("skipped")]
    comparable = [item for item in measured if "mismatched_elements" in item]
    compared_elements = sum(int(item.get("elements", 0)) for item in comparable)
    mismatched_elements = sum(int(item["mismatched_elements"]) for item in comparable)
    return {
        "compared_tensors": len(measured),
        "skipped_sketched_tensors": len(skipped),
        "mismatched_tensors": sum(not bool(item.get("passed")) for item in measured),
        "compared_elements": compared_elements,
        "mismatched_elements": mismatched_elements,
        "mismatch_ratio": (
            mismatched_elements / compared_elements if compared_elements else None
        ),
        "max_absolute_error": max(
            (float(item["max_absolute_error"]) for item in comparable),
            default=None,
        ),
        "missing_tensors": len(report.get("missing_in_candidate") or []),
        "extra_tensors": len(report.get("extra_in_candidate") or []),
    }


def compare_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict[str, Any]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    base = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if reference.shape != candidate.shape:
        if reference.size == 1 and candidate.size == 1:
            reference = reference.reshape((1,))
            candidate = candidate.reshape((1,))
            base["shape_normalized"] = "scalar-singleton"
        else:
            return {**base, "passed": False, "reason": "张量形状不同"}

    dtype = np.complex128 if np.iscomplexobj(reference) or np.iscomplexobj(candidate) else np.float64
    reference64 = reference.astype(dtype, copy=False)
    candidate64 = candidate.astype(dtype, copy=False)
    finite = np.isfinite(reference64) & np.isfinite(candidate64)
    if not bool(np.all(finite)):
        return {**base, "passed": False, "reason": "张量包含 NaN 或无穷值"}

    absolute_error = np.abs(reference64 - candidate64)
    tolerance = atol + rtol * np.abs(reference64)
    mismatch_mask = absolute_error > tolerance
    denominator = np.maximum(np.abs(reference64), np.finfo(np.float64).eps)
    relative_error = absolute_error / denominator
    reference_flat = reference64.reshape(-1)
    candidate_flat = candidate64.reshape(-1)
    norm_product = float(np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat))
    cosine_similarity = (
        float(np.vdot(reference_flat, candidate_flat).real / norm_product)
        if norm_product > 0
        else (1.0 if np.array_equal(reference64, candidate64) else 0.0)
    )
    return {
        **base,
        "passed": not bool(np.any(mismatch_mask)),
        "elements": int(reference.size),
        "mismatched_elements": int(np.count_nonzero(mismatch_mask)),
        "mismatch_ratio": float(np.mean(mismatch_mask)) if mismatch_mask.size else 0.0,
        "max_absolute_error": float(absolute_error.max(initial=0.0)),
        "mean_absolute_error": float(absolute_error.mean()) if absolute_error.size else 0.0,
        "p99_absolute_error": (
            float(np.quantile(absolute_error, 0.99)) if absolute_error.size else 0.0
        ),
        "max_relative_error": float(relative_error.max(initial=0.0)),
        "cosine_similarity": cosine_similarity,
    }
