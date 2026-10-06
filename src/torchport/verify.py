"""唯一验收判定：runnable → structural → numerical，证据缺失默认失败。"""

from __future__ import annotations

from typing import Any

from .parity import comparison_metrics

PASS, FAIL, NOT_RUN = "pass", "fail", "not_run"


def _gating_outputs(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in report.get("tensors", [])
        if item.get("name") == "output" or str(item.get("name", "")).startswith("output.")
    ]


def _structural(fp32: dict[str, Any], protocol: dict[str, Any]) -> tuple[str, list[str]]:
    problems: list[str] = []
    missing = [n for n in fp32.get("missing_in_candidate", []) if n.startswith(("output", "input"))]
    extra = [n for n in fp32.get("extra_in_candidate", []) if n.startswith("output")]
    if missing:
        problems.append(f"缺少输出/输入：{', '.join(missing)}")
    if extra:
        problems.append(f"多出输出：{', '.join(extra)}")
    allow_scalar = protocol.get("allow_scalar_singleton", False)
    check_dtype = protocol.get("check_dtype", True)
    outputs = _gating_outputs(fp32)
    if not outputs:
        problems.append("没有可比较的输出")
    for item in outputs:
        if item.get("shape_normalized") and not allow_scalar:
            problems.append(f"{item['name']} 形状不同（协议未允许标量归一化）")
        elif item.get("reference_shape") != item.get("candidate_shape") and not item.get(
            "shape_normalized"
        ):
            problems.append(f"{item['name']} 形状不同")
        if check_dtype and item.get("reference_dtype") != item.get("candidate_dtype"):
            problems.append(f"{item['name']} dtype 不同")
    return (FAIL if problems else PASS), problems


def judge(
    source_info: dict[str, Any] | None,
    target_info: dict[str, Any] | None,
    fp32: dict[str, Any] | None,
    strict: dict[str, Any] | None,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """返回 EvaluationResult；passed 只由本函数推导。"""
    protocol = protocol or {}
    result: dict[str, Any] = {
        "schema_version": 2,
        "kind": "torchport.evaluation",
        "runnable": NOT_RUN,
        "structural": NOT_RUN,
        "numerical": NOT_RUN,
        "aligned": None,
        "strict_aligned": None,
        "passed": False,
        "failure_code": None,
        "problems": [],
    }
    if not source_info or not target_info:
        result.update(runnable=FAIL, failure_code="RUN_FAILED")
        return _finish(result)
    result["runnable"] = PASS

    modes = {info.get("mode", "normal") for info in (source_info, target_info)}
    source_input = source_info.get("hashes", {}).get("input")
    target_input = target_info.get("hashes", {}).get("input")
    checkpoint = source_info.get("checkpoint", {"requested": False, "complete": True})
    state = target_info.get("state") or {}
    source_total = source_info.get("weights")
    # 双向完整：oracle 每个参数都已加载且回读一致，target 也没有 oracle 之外的参数
    complete = (
        source_total is not None
        and source_total == target_info.get("loaded")
        and state.get("target_total", source_total) == source_total
        and not state.get("extra")
        and not state.get("missing")
        and not state.get("value_mismatch")
    )
    evidence = {
        "mode": "normal" if modes == {"normal"} else "trace",
        "input_identical": bool(source_input) and source_input == target_input,
        "weights": {
            "source": source_total,
            "loaded": target_info.get("loaded"),
            "target_total": state.get("target_total"),
            "extra": state.get("extra", []),
            "missing": state.get("missing", []),
            "value_mismatch": state.get("value_mismatch", []),
            "complete": complete,
        },
        "checkpoint": checkpoint,
    }
    result["evidence"] = evidence
    if evidence["mode"] != "normal":
        result["problems"].append("诊断运行不能签发普通验收")
        result["failure_code"] = "DIAGNOSTIC_RUN"
    if not evidence["input_identical"]:
        result["problems"].append("输入摘要缺失或不一致")
        result["failure_code"] = result["failure_code"] or "INPUT_MISMATCH"
    if not evidence["weights"]["complete"] or not checkpoint.get("complete", False):
        result["problems"].append("权重或检查点加载不完整")
        result["failure_code"] = result["failure_code"] or "WEIGHT_LOAD_FAILED"

    if fp32 is None:
        result["failure_code"] = result["failure_code"] or "NOT_COMPARED"
        return _finish(result)
    result["structural"], problems = _structural(fp32, protocol)
    result["problems"].extend(problems)
    if problems:
        result["failure_code"] = result["failure_code"] or "STRUCTURE_MISMATCH"
    result["numerical"] = PASS if fp32.get("passed") else FAIL
    result["aligned"] = bool(fp32.get("passed"))
    result["strict_aligned"] = bool(strict.get("passed")) if strict is not None else None
    if result["numerical"] == FAIL:
        result["failure_code"] = result["failure_code"] or "PARITY_FAILED"
        first = fp32.get("first_gating_divergence") or {}
        item = next((t for t in fp32.get("tensors", []) if t.get("name") == first.get("name")), {})
        detail = f"首个分歧 {first.get('name')}：{first.get('reason', '数值超出容差')}"
        if "max_absolute_error" in item:
            detail += f"，max_abs={item['max_absolute_error']:.3g}"
            detail += f"，mismatch_ratio={item.get('mismatch_ratio', 0):.3g}"
        result["problems"].append(detail)
    result["profiles"] = {
        name: {
            "passed": bool(report.get("passed")),
            "first": report.get("first_gating_divergence"),
            "first_any": report.get("first_divergence"),
            "missing": report.get("missing_in_candidate", []),
            "extra": report.get("extra_in_candidate", []),
            "metrics": report.get("metrics") or comparison_metrics(report),
        }
        for name, report in (("fp32", fp32), ("strict", strict or {}))
    }
    result["mismatch"] = {
        name: profile["metrics"] for name, profile in result["profiles"].items()
    }
    result["passed"] = (
        result["failure_code"] is None
        and result["structural"] == PASS
        and result["numerical"] == PASS
    )
    return _finish(result)


def judge_repeats(results: list[dict[str, Any]], required: int) -> dict[str, Any]:
    """正式验收：预先约定的 required 次正常运行全部通过，不取最好一次。"""
    passed = len(results) >= required and all(item["passed"] for item in results)
    first_fail = next((item for item in results if not item["passed"]), None)
    return {
        "passed": passed,
        "required": required,
        "runs": len(results),
        "passed_runs": sum(item["passed"] for item in results),
        "failure_code": None if passed else (first_fail or {}).get("failure_code") or "TOO_FEW_RUNS",
    }


_NEXT = {
    "RUN_FAILED": "查看 stderr，修复目标端导入/构造错误",
    "DIAGNOSTIC_RUN": "用 mode=normal 重新运行",
    "INPUT_MISMATCH": "检查目标端是否使用 oracle 输入",
    "WEIGHT_LOAD_FAILED": "检查参数名映射和形状",
    "STRUCTURE_MISMATCH": "对齐输出容器、键、形状和 dtype",
    "PARITY_FAILED": "运行 torchport trace 定位首次分歧",
    "NOT_COMPARED": "先完成两端运行",
}


def _finish(result: dict[str, Any]) -> dict[str, Any]:
    result["status"] = "verified" if result["passed"] else "failed"
    code = result["failure_code"]
    result["next"] = _NEXT.get(code) if code else None
    result["summary"] = "通过" if result["passed"] else f"{code}: {'; '.join(result['problems'])}"
    return result
