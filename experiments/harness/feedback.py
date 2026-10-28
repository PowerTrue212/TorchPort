"""P2 三种反馈条件的裁剪规则。"""

from __future__ import annotations

from typing import Any

from .protocol import DIAGNOSTIC_FIELDS, FeedbackLevel


def filter_feedback(report: dict[str, Any], level: FeedbackLevel) -> dict[str, Any]:
    if level == FeedbackLevel.LOCALIZED:
        return report
    if level == FeedbackLevel.VERDICT:
        return {"passed": bool(report.get("passed"))}
    if report.get("kind") == "torchport.verify":
        last = (report.get("results") or [{}])[-1]
        report = {**last, **{
            key: report.get(key)
            for key in ("passed", "aligned", "strict_aligned", "mismatch", "failure_code")
            if key in report
        }}
    return diagnostic(report)


def public_run(run: dict[str, Any]) -> dict[str, Any]:
    execution = run.get("execution") or {}
    return {
        "kind": run.get("kind"),
        "run_id": run.get("run_id"),
        "ok": bool(run.get("ok")),
        "mode": run.get("mode"),
        "execution": {
            key: execution.get(key)
            for key in ("ok", "exit_code", "timeout", "seconds")
            if key in execution
        },
        "capture": {key: value for key, value in (run.get("capture") or {}).items()
                    if key in {"ok", "status", "failure_code"}},
        # 接口契约错误：只放行冻结枚举文本，两种反馈条件都可见
        "contract": _contract(run.get("contract")),
    }


def trace_summary(report: dict[str, Any]) -> dict[str, Any]:
    """限制模型可见的 trace 体积；完整报告仍留在证据文件中。"""
    def comparison(name: str) -> dict[str, Any] | None:
        value = report.get(name)
        if not isinstance(value, dict):
            return None
        return {
            key: value.get(key)
            for key in ("passed", "profile", "first_divergence", "first_gating_divergence", "metrics")
            if key in value
        }

    return {
        key: report.get(key)
        for key in ("kind", "task_id", "task_digest", "trial_id", "passed", "status", "failure_code", "stage")
        if key in report
    } | {
        "output": comparison("output"),
        "strict": comparison("strict"),
        "physical_cost": report.get("physical_cost"),
        "artifacts": report.get("artifacts"),
    }


def _contract(value: Any) -> dict[str, str] | None:
    """只放行冻结枚举里的 code 与对应文本；未知 code 或被改写的文本一律丢弃。"""
    from torchport.capture.training import CONTRACT_MESSAGES

    if not isinstance(value, dict):
        return None
    code = value.get("code")
    template = CONTRACT_MESSAGES.get(code) if isinstance(code, str) else None
    if template is None:
        return None
    message = value.get("message")
    # 唯一带参数的模板是缺失函数名；参数来自 TaskSpec，不来自候选代码
    if "{" in template:
        prefix, _, suffix = template.partition("`{name}`")
        if not (isinstance(message, str) and message.startswith(prefix)
                and message.endswith(suffix)):
            return None
    elif message != template:
        return None
    return {"code": code, "message": message}


def diagnostic(report: dict[str, Any]) -> dict[str, Any]:
    failure = str(report.get("failure_code") or "")
    codes = {
        "PARITY_FAILED": "NUMERICAL_MISMATCH",
        "STRUCTURE_MISMATCH": "STRUCTURAL_MISMATCH",
        "RUN_FAILED": "RUN_FAILED",
        "INPUT_MISMATCH": "INPUT_MISMATCH",
        "WEIGHT_LOAD_FAILED": "WEIGHT_LOAD_FAILED",
        "NOT_COMPARED": "NOT_EVALUATED",
        "TOO_FEW_RUNS": "INCOMPLETE_VERIFICATION",
        "CONTRACT_ERROR": "CONTRACT_ERROR",
    }
    code = codes.get(failure, "VERIFICATION_FAILED" if failure else None)
    structural = report.get("structural", "not_run")
    evidence = report.get("evidence") or {}
    weights = evidence.get("weights") or {}
    weight_load = "pass" if weights.get("complete") else ("fail" if weights else "not_run")
    if code == "CONTRACT_ERROR":
        stage = "contract"
    elif report.get("runnable") == "fail" or code == "RUN_FAILED":
        stage = "runnable"
    elif structural == "fail" or code == "STRUCTURAL_MISMATCH":
        stage = "structural"
    elif report.get("aligned") is False or code == "NUMERICAL_MISMATCH":
        stage = "numerical"
    elif report.get("passed"):
        stage = "passed"
    else:
        stage = "verification"
    mismatch = report.get("mismatch") or {}
    metrics = mismatch.get("fp32") or ((report.get("profiles") or {}).get("fp32") or {}).get("metrics", {})
    category = {
        "contract": "contract",
        "runnable": "runtime", "structural": "structural", "numerical": "numeric",
        "verification": "protocol", "passed": None,
    }[stage]
    result = {
        "passed": bool(report.get("passed")),
        "aligned": report.get("aligned"),
        "strict_aligned": report.get("strict_aligned"),
        "stage": stage,
        "failure_code": code,
        "category": category,
        "max_abs": metrics.get("max_absolute_error"),
        "mismatch_ratio": metrics.get("mismatch_ratio"),
        "mismatched_elements": metrics.get("mismatched_elements"),
        "mismatched_tensors": metrics.get("mismatched_tensors"),
        "structural": structural,
        "weight_load": weight_load,
        "contract": _contract(report.get("contract")),
    }
    if tuple(result) != DIAGNOSTIC_FIELDS:
        raise AssertionError("diagnostic feedback allowlist drifted")
    return result
