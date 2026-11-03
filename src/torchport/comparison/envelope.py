"""Source 变体、target 重复与环境噪声包络比较。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import SHAPE_PREFIX, _band, _diameter, _is_output, _l2, _leaf_key, _max_abs, _rel

ENVELOPE_V3 = "noise-envelope-v3"
TRAIN_REPEAT_ENVELOPE_V1 = "train-repeat-envelope-v1-pilot"


def compare_noise_calibrated(
    reference_path: str | Path,
    candidate_path: str | Path,
    fp64_path: str | Path,
    alt_path: str | Path,
    *,
    leaves: set[str] | None = None,
    k: float = 8.0,
    atol_floor: float = 1e-7,
    require_calibrated: bool = False,
    target_baseline_path: str | Path | None = None,
    drop_ustat_sum: bool = False,
) -> dict[str, Any]:
    """以 source 自身 FP32 噪声为基准判定：err(candidate, fp64) <= k*noise + atol_floor。

    noise = max(err(reference, fp64), err(reference, alt))，按 max_abs 与 p99 分别判定。
    leaves 只标记叶子模块，便于分别统计；任何已标定模块超出噪声范围都会让报告失败。
    require_calibrated 用于正式判定：缺少 fp64/alt 的 output.* 张量记为失败，不默认放过。
    """
    if k <= 0 or atol_floor < 0:
        raise ValueError("k 必须为正数，atol_floor 必须非负。")
    if drop_ustat_sum and target_baseline_path is None:
        raise ValueError("去除 ustat sum 只用于固定参考重复性包络")
    paths = [Path(p).expanduser().resolve() for p in (reference_path, candidate_path, fp64_path, alt_path)]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"证据不存在：{path}")
    baseline_path = Path(target_baseline_path).expanduser().resolve() if target_baseline_path else None
    if baseline_path is not None and not baseline_path.is_file():
        raise ValueError(f"固定目标重复性证据不存在：{baseline_path}")

    def stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        dtype = np.complex128 if np.iscomplexobj(a) or np.iscomplexobj(b) else np.float64
        diff = np.abs(a.astype(dtype) - b.astype(dtype)).reshape(-1)
        if not diff.size:
            return 0.0, 0.0
        return float(diff.max()), float(np.quantile(diff, 0.99))

    tensors: list[dict[str, Any]] = []
    missing: list[str] = []
    uncalibrated: list[str] = []
    first_divergence: dict[str, Any] | None = None
    first_gating_divergence: dict[str, Any] | None = None
    reference, candidate, fp64, alt = (np.load(p, allow_pickle=False) for p in paths)
    baseline = np.load(baseline_path, allow_pickle=False) if baseline_path is not None else None
    try:
        for position, name in enumerate(reference.files):
            if name not in candidate:
                missing.append(name)
                continue
            if name not in fp64 or name not in alt:
                uncalibrated.append(name)
                if not (require_calibrated and _is_output(name)):
                    continue
                item = {"name": name, "position": position, "passed": False, "reason": "缺少噪声标定"}
                item["gating"] = True
                tensors.append(item)
                first_divergence = first_divergence or {
                    "name": name, "position": position, "reason": item["reason"]
                }
                first_gating_divergence = first_gating_divergence or dict(first_divergence)
                continue
            if baseline is not None and _is_output(name) and name not in baseline:
                uncalibrated.append(name)
                item = {"name": name, "position": position, "passed": False, "reason": "缺少固定目标重复性证据"}
                item["gating"] = True
                tensors.append(item)
                first_divergence = first_divergence or {
                    "name": name, "position": position, "reason": item["reason"]
                }
                first_gating_divergence = first_gating_divergence or dict(first_divergence)
                continue
            ref, cand, truth, other = reference[name], candidate[name], fp64[name], alt[name]
            if not (ref.shape == cand.shape == truth.shape == other.shape):
                item = {"name": name, "position": position, "passed": False, "reason": "张量形状不同"}
            elif not all(np.all(np.isfinite(x)) for x in (ref, cand, truth, other)):
                item = {"name": name, "position": position, "passed": False, "reason": "张量包含 NaN 或无穷值"}
            elif drop_ustat_sum and name.startswith("output.ustat.") and ref.size != 3:
                item = {"name": name, "position": position, "passed": False, "reason": "ustat 必须有三个分量"}
            else:
                if drop_ustat_sum and name.startswith("output.ustat."):
                    ref, cand, truth, other = (value[:2] for value in (ref, cand, truth, other))
                cand_max, cand_p99 = stats(cand, truth)
                ref_max, ref_p99 = stats(ref, truth)
                alt_max, alt_p99 = stats(ref, other)
                noise_max, noise_p99 = max(ref_max, alt_max), max(ref_p99, alt_p99)
                if baseline is not None and _is_output(name):
                    fixed = np.asarray(baseline[name], dtype=np.float64)
                    if fixed.shape != (2,) or not np.all(np.isfinite(fixed)) or np.any(fixed < 0):
                        raise ValueError(f"固定目标重复性证据无效：{name}")
                    noise_max, noise_p99 = max(noise_max, float(fixed[0])), max(noise_p99, float(fixed[1]))
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
        for archive in (reference, candidate, fp64, alt, baseline):
            if archive is None:
                continue
            archive.close()

    gating = [item for item in tensors if item["gating"]]
    return {
        "schema_version": 1,
        "profile": TRAIN_REPEAT_ENVELOPE_V1 if baseline_path is not None else "noise-calibrated",
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

