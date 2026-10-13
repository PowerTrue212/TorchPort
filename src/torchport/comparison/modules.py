"""模块配置、隔离模块与任务指标比较。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import SHAPE_PREFIX, _band, _diameter, _l2, _rel


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
    """E2 metric-envelope-v2：任务指标差不超过 source 变体两两最大差。

    target 重复运行的波动只报告，不放宽验收上限。否则候选实现越不稳定，反而越容易通过。
    """
    if not variants or not candidates:
        raise ValueError("E2 需要至少一个 source 变体和一次 target 运行")
    items = []
    for key, value in reference.items():
        group = [float(value), *(float(v[key]) for v in variants)]
        runs = [float(c[key]) for c in candidates]
        source_noise = max(abs(a - b) for a in group for b in group)
        target_noise = max(abs(a - b) for a in runs for b in runs)
        error = max(abs(r - float(value)) for r in runs)
        limit = source_noise
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
        "profile": "metric-envelope-v2",
        "passed": all(item["passed"] for item in items) and bool(items),
        "metrics": items,
    }


def compare_isolated_v3(
    reference_path: str | Path,
    candidate_path: str | Path,
    fp64_path: str | Path,
    alt_path: str | Path,
    *,
    noise_floor: float = 0.0,
    pass_k: float = 2.0,
    floor_k: float = 3.0,
    repeat_path: str | Path | None = None,
    noise_floors: dict[str, float] | None = None,
) -> dict[str, Any]:
    """trace 模块隔离在协议 v3 下的判定基准：ratio = rel(T, R) / max(D, N_env(张量))。

    D = {R, alt, fp64} 两两最大 rel（同一份局部输入上 source 合法路径之间的差异）；N_env 是与候选无关的
    target 环境噪声下限。隔离运行的输入是 source 真实局部输入，没有上游误差传播，所以 noise_floors
    给定时按张量（即所属算子类型）取下限，缺省时退回全局 noise_floor：全局最大值来自全局归约，
    套到逐位确定的 LayerNorm 上会把 eps 这类错误完全掩盖。
    repeat_path 给定时取两次隔离运行的较大误差，并报告 target 自身波动（只报告不放宽）。
    """
    paths = [Path(p).expanduser().resolve() for p in (reference_path, candidate_path, fp64_path, alt_path)]
    if repeat_path is not None:
        paths.append(Path(repeat_path).expanduser().resolve())
    for path in paths:
        if not path.is_file():
            raise ValueError(f"证据不存在：{path}")
    archives = [np.load(p, allow_pickle=False) for p in paths]
    reference, candidate, fp64, alt = archives[:4]
    repeat = archives[4] if len(archives) > 4 else None
    tensors: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    first_marginal: dict[str, Any] | None = None
    missing: list[str] = []
    uncalibrated: list[str] = []
    try:
        candidate_names, fp64_names, alt_names = set(candidate.files), set(fp64.files), set(alt.files)
        repeat_names = set(repeat.files) if repeat is not None else set()
        for position, name in enumerate(reference.files):
            if name not in candidate_names:
                missing.append(name)
                continue
            if name not in fp64_names or name not in alt_names:
                uncalibrated.append(name)
                continue
            ref, cand = reference[name], candidate[name]
            group = [fp64[name], alt[name]]
            runs = [cand, *([repeat[name]] if name in repeat_names else [])]
            if any(x.shape != ref.shape for x in (*group, *runs)):
                item = {"name": name, "position": position, "status": "fail", "reason": "张量形状不同"}
            elif not all(np.all(np.isfinite(x)) for x in (ref, *group, *runs)):
                item = {"name": name, "position": position, "status": "fail", "reason": "张量包含 NaN 或无穷值"}
            else:
                scale = max(_l2(ref), np.finfo(np.float64).tiny)
                source_noise = _diameter([ref, *group], scale)
                error = max(_rel(run, ref, scale) for run in runs)
                floor = noise_floor if noise_floors is None else noise_floors.get(name, noise_floor)
                basis = max(source_noise, floor)
                ratio = error / basis if basis > 0 else (0.0 if error == 0 else float("inf"))
                target_noise = _diameter(runs, scale)
                item = {
                    "name": name,
                    "position": position,
                    "relative_error": error,
                    "source_noise": source_noise,
                    "target_noise": target_noise,
                    "noise_floor": floor,
                    "floor_applied": floor > source_noise,
                    "ratio": ratio,
                    "unstable": target_noise > source_noise,
                    "status": _band(ratio, pass_k, floor_k),
                }
            tensors.append(item)
            if first is None and item["status"] == "fail":
                first = {"name": name, "position": position, "reason": item.get("reason", "超出 floor_k 倍噪声包络")}
            if first_marginal is None and item["status"] == "marginal":
                first_marginal = {"name": name, "position": position, "reason": "介于 pass_k 与 floor_k 之间"}
    finally:
        for archive in archives:
            archive.close()
    ratios = [item["ratio"] for item in tensors if "ratio" in item]
    return {
        "schema_version": 1,
        "profile": "isolated-envelope-v3",
        "pass_k": pass_k,
        "floor_k": floor_k,
        "noise_floor": noise_floor,
        "per_op_floor": noise_floors is not None,
        "passed": first is None and bool(tensors),
        "first_divergence": first,
        "first_marginal": first_marginal,
        "missing_in_candidate": missing,
        "uncalibrated": uncalibrated,
        "metrics": {
            "compared_tensors": len(tensors),
            "failed_tensors": sum(item["status"] == "fail" for item in tensors),
            "marginal_tensors": sum(item["status"] == "marginal" for item in tensors),
            "max_ratio": max(ratios, default=None),
            "unstable_tensors": sum(bool(item.get("unstable")) for item in tensors),
        },
        "tensors": tensors,
    }
