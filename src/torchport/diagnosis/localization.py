"""从已有 trace 报告提取首次分歧证据。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..runtime import read_json


def localize(report: dict[str, Any]) -> dict[str, Any]:
    """从已有 trace 报告提取固定的一处首次分歧；不重新访问 oracle。"""
    if report.get("status") in {"run_failed", "evidence_missing"}:
        return {
            "kind": "torchport.localization",
            "passed": False,
            "stage": report.get("stage"),
            "failure_code": report.get("failure_code"),
            "reason": report.get("reason"),
            "first_divergence": None,
        }
    strict = report.get("strict") or {}
    output = report.get("output") or {}
    # 协议 v3：隔离比较按 max(D, N_env) 判定，固定容差会被 Jittor 后端波动先触发而指错模块
    v3 = (report.get("isolated") or {}).get("envelope_v3")
    if v3 is not None:
        return _localize_v3(report, v3)
    first = strict.get("first_divergence") or output.get("first_gating_divergence")
    tensors = strict.get("tensors") or output.get("tensors") or []
    detail = next((item for item in tensors if item.get("name") == (first or {}).get("name")), None)
    name = str((first or {}).get("name") or "")
    base, marker, count = name.rpartition("#")
    invocation = int(count) if marker and count.isdigit() else 1
    scope_name = base if marker and count.isdigit() else name
    operator = None
    if scope_name.startswith("op."):
        operator = scope_name.removeprefix("op.")
        scope_name = None
    position = (first or {}).get("position")
    previous = tensors[position - 1] if isinstance(position, int) and position > 0 else None
    current_error = (detail or {}).get("max_absolute_error")
    previous_error = (previous or {}).get("max_absolute_error")
    error_jump = (
        float(current_error) - float(previous_error)
        if current_error is not None and previous_error is not None
        else None
    )
    return {
        "kind": "torchport.localization",
        "passed": bool(report.get("passed")),
        "stage": None if report.get("passed") else ("trace" if strict.get("first_divergence") else "output"),
        "scope": scope_name,
        "invocation": invocation,
        "operator": operator,
        "first_divergence": first,
        "evidence": {"input": None, "output": detail},
        "error_jump": error_jump,
        "metrics": strict.get("metrics") or output.get("metrics"),
    }


def _localize_v3(report: dict[str, Any], v3: dict[str, Any]) -> dict[str, Any]:
    """字段与 localize 相同；误差跳变用相对误差（v3 隔离报告没有逐元素绝对误差）。

    没有 fail 时退回首个 marginal：温和错误（如 eps 写反）在隔离运行中常落在 pass_k 与 floor_k 之间。
    """
    first = v3.get("first_divergence") or v3.get("first_marginal")
    tensors = v3.get("tensors") or []
    detail = next((item for item in tensors if item.get("name") == (first or {}).get("name")), None)
    name = str((first or {}).get("name") or "")
    base, marker, count = name.rpartition("#")
    invocation = int(count) if marker and count.isdigit() else 1
    position = (first or {}).get("position")
    index = next((i for i, item in enumerate(tensors) if item.get("position") == position), None)
    previous = tensors[index - 1] if index else None
    current_error = (detail or {}).get("relative_error")
    previous_error = (previous or {}).get("relative_error")
    return {
        "kind": "torchport.localization",
        "passed": bool(report.get("passed")),
        "stage": "trace" if first else None,
        "scope": (base if marker and count.isdigit() else name) or None,
        "invocation": invocation,
        "operator": None,
        "first_divergence": first,
        "evidence": {"input": None, "output": detail},
        "error_jump": (
            float(current_error) - float(previous_error)
            if current_error is not None and previous_error is not None
            else None
        ),
        "metrics": v3.get("metrics"),
    }


def load_and_localize(path: str | Path) -> dict[str, Any]:
    return localize(read_json(Path(path).resolve()))

