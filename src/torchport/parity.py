from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

TOLERANCE_PROFILES: dict[str, tuple[float, float]] = {
    "operator-strict": (1e-5, 1e-6),
    "fp32-inference": (1e-4, 1e-4),
}
# layers.npz 中记录 sketch 前原始形状的键前缀，与 capture.sketch_layers 保持一致。
SHAPE_PREFIX = "__shape__/"


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
    k: float = 8.0,
    atol_floor: float = 1e-7,
) -> dict[str, Any]:
    """以 source 自身 FP32 噪声为基准判定：err(candidate, fp64) <= k*noise + atol_floor。

    noise = max(err(reference, fp64), err(reference, alt))，按 max_abs 与 p99 分别判定。
    leaves 只标记叶子模块，便于分别统计；任何已标定模块超出噪声范围都会让报告失败。
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
    first_divergence: dict[str, Any] | None = None
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
            if first_divergence is None and not item["passed"]:
                first_divergence = {
                    "name": name,
                    "position": position,
                    "reason": item.get("reason", "超出 source 噪声底"),
                }
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
        "passed": first_divergence is None and bool(tensors),
        "first_divergence": first_divergence,
        "first_gating_divergence": first_gating_divergence,
        "missing_in_candidate": missing,
        "uncalibrated": uncalibrated,
        "metrics": {
            "compared_tensors": len(tensors),
            "gating_tensors": len(gating),
            "failed_tensors": sum(not item["passed"] for item in tensors),
            "failed_gating_tensors": sum(not item["passed"] for item in gating),
            "failed_reported_tensors": sum(not item["passed"] for item in tensors if not item["gating"]),
            "max_ratio": max(
                (max(item["ratio_max"], item["ratio_p99"]) for item in tensors if "ratio_max" in item),
                default=None,
            ),
            "max_gating_ratio": max(
                (max(item["ratio_max"], item["ratio_p99"]) for item in gating if "ratio_max" in item),
                default=None,
            ),
        },
        "tensors": tensors,
    }


def _l2(x: np.ndarray) -> float:
    # 不走 np.linalg.norm：它调用多线程 BLAS dot，在 65536 维 sketch 上线程调度开销
    # 比计算本身大约 4500 倍（48 线程实测 7e-2 s 对 1.6e-5 s）。ufunc 求和不进 BLAS。
    flat = x.astype(np.float64, copy=False).reshape(-1)
    return float(np.sqrt(np.sum(flat * flat)))


def _rel(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    return _l2(a.astype(np.float64) - b.astype(np.float64)) / scale


def _diameter(arrays: list[np.ndarray], scale: float) -> float:
    """一组合法运行两两之间的最大相对距离；少于两份时为 0。"""
    return max(
        (_rel(arrays[i], arrays[j], scale) for i in range(len(arrays)) for j in range(i + 1, len(arrays))),
        default=0.0,
    )


def compare_envelope(
    reference_path: str | Path,
    variant_paths: list[str | Path],
    candidate_path: str | Path,
    peer_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """E1：逐层相对误差不超过 source 合法变体的噪声包络加 target 自身波动。

    rel(x, y) = ||x - y|| / ||R||。对每个双方都有的张量：
      D = {R, V1..Vn} 两两最大 rel（PyTorch 合法运行之间能相差多远）
      N = {T, peers} 两两最大 rel（target 自身运行间波动）
      通过条件：rel(T, R) <= D + N。没有倍数参数；ratio = rel(T, R) / (D + N) 仅作报告。
    名称只在一端出现的张量只报告不判定（命名自由不算实现错误），但 output.* 必须齐全。
    """
    paths = [Path(p).expanduser().resolve() for p in (reference_path, candidate_path, *variant_paths)]
    peers = [Path(p).expanduser().resolve() for p in (peer_paths or [])]
    for path in (*paths, *peers):
        if not path.is_file():
            raise ValueError(f"证据不存在：{path}")
    if not variant_paths:
        raise ValueError("noise-envelope 至少需要一个 source 合法变体")
    archives = [np.load(p, allow_pickle=False) for p in (*paths, *peers)]
    reference, candidate = archives[0], archives[1]
    tensors: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    try:
        def tensor_names(archive: Any) -> list[str]:
            # `__shape__/<name>` 是 sketch 保存的原始形状，只用于形状核对，不作为张量比较。
            return [
                n for n in archive.files
                if not (n == "input" or n.startswith(("input.", SHAPE_PREFIX)))
            ]

        # NpzFile 每次下标访问都会重新读出成员；成员名集合只建一次，每个张量每份证据只读一次。
        members = [set(archive.files) for archive in archives]

        def load(index: int, name: str) -> tuple[np.ndarray, tuple[int, ...]] | None:
            if name not in members[index]:
                return None
            value = archives[index][name]
            key = SHAPE_PREFIX + name
            shape = (
                tuple(int(x) for x in archives[index][key]) if key in members[index] else value.shape
            )
            return value, shape

        names = tensor_names(reference)
        missing = [n for n in names if n not in members[1]]
        extra = [n for n in tensor_names(candidate) if n not in members[0]]
        variant_index = range(2, 2 + len(variant_paths))
        peer_index = range(2 + len(variant_paths), len(archives))
        for position, name in enumerate(names):
            if name in missing:
                if name == "output" or name.startswith("output."):
                    item = {"name": name, "position": position, "passed": False, "reason": "候选缺少输出"}
                    tensors.append(item)
                    first = first or {"name": name, "position": position, "reason": item["reason"]}
                continue
            loaded = {i: load(i, name) for i in range(len(archives))}
            present = [entry for entry in loaded.values() if entry is not None]
            ref, cand = loaded[0][0], loaded[1][0]
            group = [loaded[i][0] for i in variant_index if loaded[i] is not None]
            others = [loaded[i][0] for i in peer_index if loaded[i] is not None]
            # 原始形状（sketch 前）与存储形状（sketch 后）都必须在所有运行间一致。
            original_shapes = {shape for _, shape in present}
            stored_shapes = {ref.shape, cand.shape, *(x.shape for x in group), *(x.shape for x in others)}
            is_output = name == "output" or name.startswith("output.")
            if len(original_shapes) != 1 or len(stored_shapes) != 1:
                item = {"name": name, "position": position, "passed": False, "reason": "张量形状不同"}
            elif is_output and ref.dtype != cand.dtype:
                item = {"name": name, "position": position, "passed": False, "reason": "输出 dtype 不同"}
            elif not all(np.all(np.isfinite(x)) for x in (ref, cand, *group, *others)):
                item = {"name": name, "position": position, "passed": False, "reason": "张量包含 NaN 或无穷值"}
            elif not group:
                item = {"name": name, "position": position, "passed": False, "reason": "缺少 source 变体"}
            else:
                scale = max(_l2(ref), np.finfo(np.float64).tiny)
                error = _rel(cand, ref, scale)
                source_noise = _diameter([ref, *group], scale)
                target_noise = _diameter([cand, *others], scale)
                limit = source_noise + target_noise
                item = {
                    "name": name,
                    "position": position,
                    "elements": int(ref.size),
                    "relative_error": error,
                    "source_noise": source_noise,
                    "target_noise": target_noise,
                    "limit": limit,
                    "ratio": error / limit if limit > 0 else (0.0 if error == 0 else float("inf")),
                    # 每个变体相对 R 的距离，便于离线按变体子集重算包络
                    "variant_errors": [_rel(v, ref, scale) for v in group],
                    "passed": error <= limit,
                }
            tensors.append(item)
            if first is None and not item["passed"]:
                first = {"name": name, "position": position, "reason": item.get("reason", "超出噪声包络")}
    finally:
        for archive in archives:
            archive.close()
    ratios = [item["ratio"] for item in tensors if "ratio" in item]
    return {
        "schema_version": 1,
        "profile": "noise-envelope",
        "variants": len(variant_paths),
        "target_peers": len(peers),
        "passed": first is None and any(item.get("name", "").startswith("output") for item in tensors),
        "first_divergence": first,
        "missing_in_candidate": missing,
        "extra_in_candidate": extra,
        "metrics": {
            "compared_tensors": len(tensors),
            "failed_tensors": sum(not item["passed"] for item in tensors),
            "max_ratio": max(ratios, default=None),
            "missing_tensors": len(missing),
            "extra_tensors": len(extra),
        },
        "tensors": tensors,
    }


ENVELOPE_V3 = "noise-envelope-v3"


def _is_output(name: str) -> bool:
    return name == "output" or name.startswith("output.")


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), initial=0.0))


def _band(ratio: float, pass_k: float, floor_k: float) -> str:
    return "pass" if ratio <= pass_k else ("marginal" if ratio <= floor_k else "fail")


def compare_envelope_v3(
    reference_path: str | Path,
    variant_paths: list[str | Path],
    candidate_paths: list[str | Path],
    *,
    pass_k: float = 2.0,
    floor_k: float = 3.0,
    batch_variants: tuple[int, ...] = (),
    noise_floor: float = 0.0,
) -> dict[str, Any]:
    """E1 协议 v3：每次 target 运行都必须落在 k 倍 source 合法变体包络之内。

    对每个张量：D = {R, 有效变体} 两两最大 rel；ratio = max_i rel(T_i, R) / max(D, noise_floor)。
    noise_floor 是冻结在协议里的 target 环境噪声下限（按算子实测的运行间波动取最大值），
    与候选无关，属于可调超参。
    与 v2 的区别：
      - 所有 target 运行逐次参与判定（取最大），结果与运行顺序无关；
      - target 自身波动 N 只报告（N > D 记为 unstable），不放宽上限；
      - 判定分三档：ratio <= pass_k 通过，<= floor_k 为 marginal（可提交、需看 E2），否则失败；
      - output.* 不压缩，另按逐元素最大绝对误差 / 变体最大绝对差计比值，取两者较大；
      - 批量变体在某张量上比其他变体大 floor_k 倍以上时视为模型批内相关，从 D 中剔除并报告；
      - 报告覆盖率与每个张量实际产生差异的变体数（少于 2 记为 thin）。
    """
    if not variant_paths:
        raise ValueError("noise-envelope-v3 至少需要一个 source 合法变体")
    if not candidate_paths:
        raise ValueError("noise-envelope-v3 至少需要一次 target 运行")
    paths = [Path(p).expanduser().resolve() for p in (reference_path, *variant_paths, *candidate_paths)]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"证据不存在：{path}")
    archives = [np.load(p, allow_pickle=False) for p in paths]
    members = [set(archive.files) for archive in archives]
    variant_index = list(range(1, 1 + len(variant_paths)))
    candidate_index = list(range(1 + len(variant_paths), len(archives)))
    batch = {variant_index[i] for i in batch_variants if 0 <= i < len(variant_index)}

    def names_of(index: int) -> list[str]:
        return [
            n for n in archives[index].files
            if not (n == "input" or n.startswith(("input.", SHAPE_PREFIX)))
        ]

    def load(index: int, name: str) -> tuple[np.ndarray, tuple[int, ...]] | None:
        if name not in members[index]:
            return None
        value = archives[index][name]
        key = SHAPE_PREFIX + name
        shape = tuple(int(x) for x in archives[index][key]) if key in members[index] else value.shape
        return value, shape

    tensors: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    first_marginal: dict[str, Any] | None = None
    try:
        names = names_of(0)
        missing = [n for n in names if any(n not in members[i] for i in candidate_index)]
        extra = sorted({n for i in candidate_index for n in names_of(i) if n not in members[0]})
        for position, name in enumerate(names):
            is_output = _is_output(name)
            if name in missing:
                if is_output:
                    item = {"name": name, "position": position, "status": "fail", "reason": "候选缺少输出"}
                    tensors.append(item)
                    first = first or {"name": name, "position": position, "reason": item["reason"]}
                continue
            ref, ref_shape = load(0, name)
            cands = [load(i, name) for i in candidate_index]
            # 变体形状与参考不同（例如批量变体拆不开）只少一份证据，不判候选失败
            variants = {
                i: entry for i in variant_index
                if (entry := load(i, name)) is not None and entry[1] == ref_shape and entry[0].shape == ref.shape
            }
            shapes = {ref_shape, *(s for _, s in cands)}
            stored = {ref.shape, *(c.shape for c, _ in cands)}
            values = [ref, *(v for v, _ in variants.values()), *(c for c, _ in cands)]
            reason = None
            if len(shapes) != 1 or len(stored) != 1:
                reason = "张量形状不同"
            elif is_output and any(c.dtype != ref.dtype for c, _ in cands):
                reason = "输出 dtype 不同"
            elif not all(np.all(np.isfinite(x)) for x in values):
                reason = "张量包含 NaN 或无穷值"
            elif not variants:
                reason = "缺少 source 变体"
            if reason:
                item = {"name": name, "position": position, "status": "fail", "reason": reason}
                tensors.append(item)
                first = first or {"name": name, "position": position, "reason": reason}
                continue
            scale = max(_l2(ref), np.finfo(np.float64).tiny)
            distance = {i: _rel(v, ref, scale) for i, (v, _) in variants.items()}
            others = [d for i, d in distance.items() if i not in batch]
            suspect = [
                i for i in batch
                if i in distance and others and max(others) > 0 and distance[i] > floor_k * max(others)
            ]
            included = [i for i in variants if i not in suspect]
            group = [variants[i][0] for i in included]
            source_noise = _diameter([ref, *group], scale)
            run_errors = [_rel(c, ref, scale) for c, _ in cands]
            error = max(run_errors)
            target_noise = _diameter([c for c, _ in cands], scale)
            # 上限基数取 D 与 target 环境噪声下限的较大者；下限与候选无关，不随候选自身波动变宽
            basis = max(source_noise, noise_floor)
            ratio = error / basis if basis > 0 else (0.0 if error == 0 else float("inf"))
            item = {
                "name": name,
                "position": position,
                "elements": int(np.prod(ref_shape)) if ref_shape else 1,
                "relative_error": error,
                "run_errors": run_errors,
                "source_noise": source_noise,
                "target_noise": target_noise,
                "floor_applied": noise_floor > source_noise,
                "limit": pass_k * basis,
                "l2_ratio": ratio,
                "variant_errors": [distance.get(i) for i in variant_index],
                "effective_variants": sum(distance[i] > 0 for i in included),
                "batch_excluded": len(suspect),
                "unstable": target_noise > source_noise,
            }
            exact = int(ref.size) == item["elements"]
            if is_output and exact:
                # 输出逐元素：最大绝对误差相对 source 变体间最大绝对差
                source_abs = max((_max_abs(v, ref) for v in group), default=0.0)
                target_abs = max(_max_abs(c, ref) for c, _ in cands)
                abs_ratio = (
                    target_abs / source_abs if source_abs > 0 else (0.0 if target_abs == 0 else float("inf"))
                )
                item.update(max_abs_error=target_abs, source_max_abs=source_abs, abs_ratio=abs_ratio)
                ratio = max(ratio, abs_ratio)
            item["ratio"] = ratio
            item["status"] = _band(ratio, pass_k, floor_k)
            tensors.append(item)
            if item["status"] == "fail" and first is None:
                first = {"name": name, "position": position, "reason": "超出 floor_k 倍噪声包络"}
            if item["status"] == "marginal" and first_marginal is None:
                first_marginal = {"name": name, "position": position, "reason": "介于 pass_k 与 floor_k 之间"}
    finally:
        for archive in archives:
            archive.close()
    has_output = any(_is_output(item["name"]) and "ratio" in item for item in tensors)
    statuses = {item["status"] for item in tensors}
    if first is not None or "fail" in statuses or not has_output:
        status = "fail"
    elif "marginal" in statuses:
        status = "marginal"
    else:
        status = "pass"
    ratios = [item["ratio"] for item in tensors if "ratio" in item]
    compared = len(tensors)
    return {
        "schema_version": 1,
        "profile": ENVELOPE_V3,
        "pass_k": pass_k,
        "floor_k": floor_k,
        "noise_floor": noise_floor,
        "variants": len(variant_paths),
        "target_runs": len(candidate_paths),
        "status": status,
        "passed": status == "pass",
        "first_divergence": first if status == "fail" else None,
        "first_marginal": first_marginal,
        "missing_in_candidate": missing,
        "extra_in_candidate": extra,
        "metrics": {
            "compared_tensors": compared,
            "failed_tensors": sum(item["status"] == "fail" for item in tensors),
            "marginal_tensors": sum(item["status"] == "marginal" for item in tensors),
            "max_ratio": max(ratios, default=None),
            "missing_tensors": len(missing),
            "extra_tensors": len(extra),
            "coverage": compared / len(names) if names else 0.0,
            "unstable_tensors": sum(bool(item.get("unstable")) for item in tensors),
            "thin_envelope_tensors": sum(item.get("effective_variants", 2) < 2 for item in tensors),
            "batch_excluded_tensors": sum(bool(item.get("batch_excluded")) for item in tensors),
            "floor_applied_tensors": sum(bool(item.get("floor_applied")) for item in tensors),
        },
        "tensors": tensors,
    }


def _same_config(a: Any, b: Any) -> bool:
    # int 与等值整数列表视为相同（PyTorch Conv 存 (1, 1)，部分实现存 1）
    if isinstance(a, list) and not isinstance(b, list):
        return all(_same_config(item, b) for item in a)
    if isinstance(b, list) and not isinstance(a, list):
        return all(_same_config(a, item) for item in b)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same_config(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
            return a == b
        return abs(float(a) - float(b)) <= 1e-12 * max(abs(float(a)), abs(float(b)), 1e-300)
    return a == b


def compare_module_config(
    source: dict[str, dict[str, Any]],
    target: dict[str, dict[str, Any]],
    called: set[str] | None = None,
) -> dict[str, Any]:
    """同名模块的构造超参逐键比较；只比较两端都记录了的键，只作诊断证据，不参与判定。

    called 给定时（source 前向中实际被调用的模块名），未被调用模块上的差异标为 used=False：
    它们是死代码里的差异，不影响数值。
    """
    mismatches = []
    compared = 0
    for name, entry in source.items():
        other = target.get(name)
        if other is None:
            continue
        for key, value in entry.items():
            if key == "class" or key not in other:
                continue
            compared += 1
            if not _same_config(value, other[key]):
                mismatches.append({
                    "module": name,
                    "key": key,
                    "source": value,
                    "target": other[key],
                    "source_class": entry.get("class"),
                    "target_class": other.get("class"),
                    "used": None if called is None else name in called,
                })
    return {
        "schema_version": 1,
        "kind": "torchport.module_config",
        "compared_keys": compared,
        "common_modules": len(set(source) & set(target)),
        "source_only_modules": len(set(source) - set(target)),
        "used_mismatches": sum(item["used"] is not False for item in mismatches),
        "mismatches": mismatches,
    }


def called_modules(layers_path: str | Path) -> set[str]:
    """从 layers 证据的张量名还原前向中实际被调用的模块名（去掉 `#n` 调用序号与输出下标）。"""
    with np.load(layers_path, allow_pickle=False) as archive:
        names = [n for n in archive.files if not n.startswith(SHAPE_PREFIX)]
    called: set[str] = set()
    for name in names:
        base = name.split("#", 1)[0]
        called.add(base)
        # 多输出模块的张量名形如 `<module>.0`；把去掉末段数字的前缀也算作被调用
        head, _, tail = base.rpartition(".")
        if head and tail.isdigit():
            called.add(head)
    return called


def compare_metric_envelope(
    reference: dict[str, float],
    variants: list[dict[str, float]],
    candidates: list[dict[str, float]],
) -> dict[str, Any]:
    """E2：任务指标与 reference 的差不超过 source 变体两两最大差 + target 重复两两最大差。"""
    if not variants or not candidates:
        raise ValueError("E2 需要至少一个 source 变体和一次 target 运行")
    items = []
    for key, value in reference.items():
        group = [float(value), *(float(v[key]) for v in variants)]
        runs = [float(c[key]) for c in candidates]
        source_noise = max(abs(a - b) for a in group for b in group)
        target_noise = max(abs(a - b) for a in runs for b in runs)
        error = max(abs(r - float(value)) for r in runs)
        limit = source_noise + target_noise
        items.append({
            "metric": key,
            "reference": float(value),
            "candidate": runs,
            "error": error,
            "source_noise": source_noise,
            "target_noise": target_noise,
            "limit": limit,
            "ratio": error / limit if limit > 0 else (0.0 if error == 0 else float("inf")),
            "passed": error <= limit,
        })
    return {
        "schema_version": 1,
        "profile": "metric-envelope",
        "passed": all(item["passed"] for item in items) and bool(items),
        "metrics": items,
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
