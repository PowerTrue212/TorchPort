from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

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
        reference_names = list(reference.files)
        candidate_names = list(candidate.files)
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
                item = compare_arrays(
                    reference[name],
                    candidate[name],
                    rtol=rtol,
                    atol=atol,
                )
                item.update({"name": name, "position": position})
            item["gating"] = gating
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
        "first_divergence": first_divergence,
        "first_gating_divergence": first_gating_divergence,
        "missing_in_candidate": missing,
        "extra_in_candidate": extra,
        "tensors": comparisons,
    }
    report["metrics"] = comparison_metrics(report)
    return report


def compare_noise_calibrated(
    reference_path: str | Path,
    candidate_path: str | Path,
    fp64_path: str | Path,
    alt_path: str | Path,
    *,
    leaves: set[str] | None = None,
    k: float = 4.0,
    atol_floor: float = 1e-7,
) -> dict[str, Any]:
    """以 source 自身 FP32 噪声为基准判定：err(candidate, fp64) <= k*noise + atol_floor。

    noise = max(err(reference, fp64), err(reference, alt))，按 max_abs 与 p99 分别判定。
    leaves 给定时只有叶子模块参与判定，组合模块只报告。
    """
    if k <= 0 or atol_floor < 0:
        raise ValueError("k 必须为正数，atol_floor 必须非负。")
    paths = [Path(p).expanduser().resolve() for p in (reference_path, candidate_path, fp64_path, alt_path)]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"证据不存在：{path}")

    def stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64)).reshape(-1)
        if not diff.size:
            return 0.0, 0.0
        return float(diff.max()), float(np.quantile(diff, 0.99))

    tensors: list[dict[str, Any]] = []
    missing: list[str] = []
    uncalibrated: list[str] = []
    first_gating_divergence: dict[str, Any] | None = None
    reference, candidate, fp64, alt = (np.load(p, allow_pickle=False) for p in paths)
    try:
        for position, name in enumerate(reference.files):
            if name not in candidate:
                missing.append(name)
                continue
            if name not in fp64 or name not in alt:
                uncalibrated.append(name)
                continue
            ref, cand, truth, other = reference[name], candidate[name], fp64[name], alt[name]
            if not (ref.shape == cand.shape == truth.shape == other.shape):
                item = {"name": name, "position": position, "passed": False, "reason": "张量形状不同"}
            elif not all(np.all(np.isfinite(x)) for x in (ref, cand, truth, other)):
                item = {"name": name, "position": position, "passed": False, "reason": "张量包含 NaN 或无穷值"}
            else:
                cand_max, cand_p99 = stats(cand, truth)
                ref_max, ref_p99 = stats(ref, truth)
                alt_max, alt_p99 = stats(ref, other)
                noise_max, noise_p99 = max(ref_max, alt_max), max(ref_p99, alt_p99)
                limit_max, limit_p99 = k * noise_max + atol_floor, k * noise_p99 + atol_floor
                item = {
                    "name": name,
                    "position": position,
                    "elements": int(ref.size),
                    "candidate_max_abs": cand_max,
                    "candidate_p99_abs": cand_p99,
                    "noise_max_abs": noise_max,
                    "noise_p99_abs": noise_p99,
                    "ratio_max": cand_max / max(noise_max, atol_floor),
                    "ratio_p99": cand_p99 / max(noise_p99, atol_floor),
                    "passed": cand_max <= limit_max and cand_p99 <= limit_p99,
                }
            item["gating"] = leaves is None or _leaf_key(name, leaves)
            tensors.append(item)
            if first_gating_divergence is None and item["gating"] and not item["passed"]:
                first_gating_divergence = {
                    "name": name,
                    "position": position,
                    "reason": item.get("reason", "超出 source 噪声底"),
                }
    finally:
        for archive in (reference, candidate, fp64, alt):
            archive.close()

    gating = [item for item in tensors if item["gating"]]
    return {
        "schema_version": 1,
        "profile": "noise-calibrated",
        "k": k,
        "atol_floor": atol_floor,
        "passed": first_gating_divergence is None and bool(gating),
        "first_gating_divergence": first_gating_divergence,
        "missing_in_candidate": missing,
        "uncalibrated": uncalibrated,
        "metrics": {
            "compared_tensors": len(tensors),
            "gating_tensors": len(gating),
            "failed_gating_tensors": sum(not item["passed"] for item in gating),
            "failed_reported_tensors": sum(not item["passed"] for item in tensors if not item["gating"]),
            "max_gating_ratio": max(
                (max(item["ratio_max"], item["ratio_p99"]) for item in gating if "ratio_max" in item),
                default=None,
            ),
        },
        "tensors": tensors,
    }


def _leaf_key(name: str, leaves: set[str]) -> bool:
    """张量名形如 `<module>`、`<module>#2`、`<module>.0`；去掉调用序号与输出下标后匹配模块名。"""
    base = name.split("#", 1)[0]
    while base:
        if base in leaves:
            return True
        if "." not in base:
            return False
        head, tail = base.rsplit(".", 1)
        if not tail.isdigit():
            return False
        base = head
    return False


def comparison_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """把逐张量证据归约成稳定、可聚合的计数；结构不一致不伪造元素误差。"""
    tensors = list(report.get("tensors") or [])
    comparable = [item for item in tensors if "mismatched_elements" in item]
    compared_elements = sum(int(item.get("elements", 0)) for item in comparable)
    mismatched_elements = sum(int(item["mismatched_elements"]) for item in comparable)
    return {
        "compared_tensors": len(tensors),
        "mismatched_tensors": sum(not bool(item.get("passed")) for item in tensors),
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

    reference64 = reference.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
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
        float(np.dot(reference_flat, candidate_flat) / norm_product)
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
