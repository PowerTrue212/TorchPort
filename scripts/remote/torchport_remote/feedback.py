"""远端执行结果的反馈裁剪。"""

from __future__ import annotations

import json
from typing import Any


def diagnostic_view(report: dict[str, Any]) -> dict[str, Any]:
    """只保留阶段、失败类别、聚合数值、结构和权重状态。"""
    result = (report.get("results") or [report])[-1]
    evidence = result.get("evidence") or {}
    weights = evidence.get("weights") or {}
    metrics = (report.get("e1") or {}).get("metrics") or {}
    if result.get("runnable") == "fail":
        stage = "runnable"
    elif result.get("structural") == "fail":
        stage = "structural"
    elif result.get("numerical") == "fail":
        stage = "numerical"
    elif report.get("passed"):
        stage = "passed"
    else:
        stage = "verification"
    return {
        "passed": bool(report.get("passed")),
        "e1_status": report.get("e1_status"),
        "stage": stage,
        "failure_code": report.get("failure_code"),
        "structural": result.get("structural", "not_run"),
        "weight_load": "pass" if weights.get("complete") else ("fail" if weights else "not_run"),
        "compared_tensors": metrics.get("compared_tensors"),
        "failed_tensors": metrics.get("failed_tensors"),
        "max_ratio": metrics.get("max_ratio"),
    }


def filter_output(command: str, out: str, feedback: str) -> str:
    if feedback == "localized" or command not in {"verify", "evaluate"}:
        return out
    try:
        report = json.loads(out)
    except json.JSONDecodeError:
        return json.dumps({"passed": False, "stage": "verification", "failure_code": "UNPARSEABLE"})
    return json.dumps(diagnostic_view(report), ensure_ascii=False, indent=2)
