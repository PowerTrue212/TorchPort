"""Pilot-only arithmetic for the four Benchmark v1 metrics.

This module does not run a model and does not make an official acceptance
decision. It consumes controller-owned evidence. Synthetic task metrics and
Agent events are permitted only because the output is explicitly a smoke test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须是数字")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} 必须是有限{'正' if positive else ''}数")
    return result


def _r_e(trial: dict[str, Any], official: dict[str, Any]) -> dict[str, Any]:
    required = ("inference", "training")
    if any(not isinstance(official.get(name), dict) for name in required):
        raise ValueError("隐藏验收缺少 inference 或 training 报告")
    verify_passed = all(official[name].get("passed") is True for name in required)
    if trial.get("passed") is not verify_passed:
        raise ValueError("trial.result 的 passed 与隐藏验收报告不一致")
    runnable = {}
    for name in required:
        repeats = official[name].get("results")
        if not isinstance(repeats, list) or not repeats:
            raise ValueError(f"{name} 缺少逐次运行证据")
        runnable[name] = all(item.get("runnable") == "pass" for item in repeats)
    return {
        "value": int(all(runnable.values())),
        "status": "measured_technical",
        "inference_runnable": runnable["inference"],
        "training_runnable": runnable["training"],
        "verify_passed": verify_passed,
    }


def _b_l(layer: dict[str, Any]) -> dict[str, Any]:
    if layer.get("status") != "exploratory_not_formal":
        raise ValueError("试跑的 B_l 必须标记为 exploratory_not_formal")
    count = layer.get("checkpoint_count")
    rows = layer.get("rows")
    score = _number(layer.get("score"), "B_l.score")
    if not isinstance(count, int) or count < 1 or not isinstance(rows, list) or len(rows) != count:
        raise ValueError("B_l 检查点数量与逐点数据不一致")
    if not 0 <= score <= 100:
        raise ValueError("B_l.score 必须在 0 到 100 之间")
    scores = [_number(row.get("score"), "B_l.rows.score") for row in rows]
    if any(not 0 <= item <= 1 for item in scores):
        raise ValueError("B_l 逐点得分必须在 0 到 1 之间")
    recomputed = 100 * sum(scores) / count
    if not math.isclose(score, recomputed, rel_tol=0, abs_tol=1e-9):
        raise ValueError("B_l 总分与逐点数据不一致")
    return {"value": score, "status": "exploratory", "checkpoints": count}


def _p_t(task: dict[str, Any]) -> dict[str, Any]:
    if task.get("provenance") != "synthetic_smoke":
        raise ValueError("试跑 P_t 只能读取明确标记的合成样例")
    metrics = task.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("P_t 至少需要一项任务指标")
    names: set[str] = set()
    rows = []
    for item in metrics:
        if not isinstance(item, dict):
            raise ValueError("P_t 指标必须是对象")
        name = item.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("P_t 指标名缺失或重复")
        names.add(name)
        source = _number(item.get("source"), f"{name}.source")
        target = _number(item.get("target"), f"{name}.target")
        tau = _number(item.get("tau"), f"{name}.tau", positive=True)
        ratio = abs(target - source) / tau
        rows.append({
            "name": name,
            "source": source,
            "target": target,
            "tau": tau,
            "g": ratio,
            "score": 1 / max(1, ratio),
        })
    value = 100 * sum(row["score"] for row in rows) / len(rows)
    return {
        "value": value,
        "status": "illustrative_synthetic",
        "s": int(all(row["g"] <= 1 for row in rows)),
        "metrics": rows,
    }


def _agent_cost(events_path: Path, baselines: dict[str, Any], s: int) -> dict[str, Any]:
    if baselines.get("provenance") != "synthetic_smoke":
        raise ValueError("试跑 E_m 只能读取明确标记的合成样例")
    normalizers = {
        name: _number(baselines.get(name), name, positive=True)
        for name in ("T0", "N0", "K0", "F0")
    }
    events = []
    for index, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Agent 事件必须是对象：{index}")
        events.append(item)
    kinds = [item.get("event") for item in events]
    if kinds.count("agent_session_started") != 1 or kinds.count("agent_session_finished") != 1:
        raise ValueError("Agent 事件缺少唯一的会话开始或结束标记")

    intervals: list[tuple[float, float]] = []
    tokens = 0
    responses = 0
    tools = 0
    revisions: set[str] = set()
    checks: dict[str, bool] = {}
    for item in events:
        kind = item.get("event")
        if kind == "work_interval" and item.get("category") == "migration":
            start = _number(item.get("started_at"), "work_interval.started_at")
            end = _number(item.get("finished_at"), "work_interval.finished_at")
            if end < start:
                raise ValueError("迁移时间区间的结束时间早于开始时间")
            intervals.append((start, end))
        elif kind == "model_response":
            usage = item.get("usage") or {}
            total = _number(usage.get("total_tokens"), "usage.total_tokens")
            if not total.is_integer() or total < 0:
                raise ValueError("usage.total_tokens 必须是非负整数")
            tokens += int(total)
            responses += 1
        elif kind == "agent_tool_call":
            tools += 1
        elif kind == "candidate_revision":
            digest = item.get("digest")
            if not isinstance(digest, str) or not digest:
                raise ValueError("candidate_revision.digest 缺失")
            revisions.add(digest)
        elif kind == "revision_check":
            digest = item.get("digest")
            if not isinstance(digest, str) or not digest or not isinstance(item.get("passed"), bool):
                raise ValueError("revision_check 缺少 digest 或 passed")
            if digest in checks and checks[digest] is not item["passed"]:
                raise ValueError("同一候选版本的统一检查结果冲突")
            checks[digest] = item["passed"]
    if not intervals or responses == 0 or not revisions or set(checks) != revisions:
        raise ValueError("Agent 成本事件不完整")
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("迁移时间区间发生重叠")
    t = sum(end - start for start, end in intervals)
    f = sum(not checks[digest] for digest in revisions)
    c = (0.4 * t / normalizers["T0"] + 0.4 * tokens / normalizers["N0"]
         + 0.1 * tools / normalizers["K0"] + 0.1 * f / normalizers["F0"])
    return {
        "value": 100 * s / (1 + c),
        "status": "illustrative_synthetic",
        "T": t,
        "N": tokens,
        "K": tools,
        "F": f,
        "C": c,
        "normalizers": normalizers,
    }


def score_smoke(
    *,
    trial_result: Path,
    official_report: Path,
    layer_score: Path | None,
    task_metrics: Path | None,
    agent_events: Path | None,
    normalizers: Path | None,
) -> dict[str, Any]:
    """Score a pilot fixture. Never return a formal success claim."""
    trial = _read_json(trial_result)
    if trial.get("claim_status") != "pilot_only" or trial.get("valid") is not False:
        raise ValueError("该入口只接受 valid=false 的 pilot_only trial")
    official = _read_json(official_report)
    metrics = {
        "R_e": _r_e(trial, official),
        "B_l": {"value": None, "status": "undefined", "reason": "layer_evidence_missing"},
        "P_t": {"value": None, "status": "undefined", "reason": "task_metrics_missing"},
        "E_m": {"value": None, "status": "undefined", "reason": "agent_cost_missing"},
    }
    evidence = {
        "trial_result": {"path": str(trial_result), "sha256": _sha256(trial_result)},
        "official_report": {"path": str(official_report), "sha256": _sha256(official_report)},
    }
    if layer_score is not None:
        metrics["B_l"] = _b_l(_read_json(layer_score))
        evidence["layer_score"] = {"path": str(layer_score), "sha256": _sha256(layer_score)}
    if task_metrics is not None:
        metrics["P_t"] = _p_t(_read_json(task_metrics))
        evidence["task_metrics"] = {"path": str(task_metrics), "sha256": _sha256(task_metrics)}
    if agent_events is not None or normalizers is not None:
        if agent_events is None or normalizers is None:
            raise ValueError("E_m 必须同时提供 Agent 事件与归一化常数")
        if metrics["P_t"]["value"] is None:
            metrics["E_m"]["reason"] = "task_success_undefined"
        else:
            metrics["E_m"] = _agent_cost(
                agent_events, _read_json(normalizers), metrics["P_t"]["s"]
            )
        evidence["agent_events"] = {"path": str(agent_events), "sha256": _sha256(agent_events)}
        evidence["normalizers"] = {"path": str(normalizers), "sha256": _sha256(normalizers)}
    return {
        "schema_version": 1,
        "kind": "benchmark-v1-metric-pipeline-smoke",
        "trial_id": trial.get("trial_id"),
        "claim_status": "pilot_only",
        "valid": False,
        "formal_eligible": False,
        "limitations": [
            "R_e 仅是冻结候选的运行性，不代表正式 trial 有效",
            "B_l 使用事后选定的检查点和容差",
            "P_t 若有数值，其输入是合成样例，不是 E2E 任务结果",
            "E_m 若有数值，其输入是合成 Agent 事件，不是 r1 Agent 成本",
        ],
        "metrics": metrics,
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-result", type=Path, required=True)
    parser.add_argument("--official-report", type=Path, required=True)
    parser.add_argument("--layer-score", type=Path)
    parser.add_argument("--task-metrics", type=Path)
    parser.add_argument("--agent-events", type=Path)
    parser.add_argument("--normalizers", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score_smoke(
        trial_result=args.trial_result,
        official_report=args.official_report,
        layer_score=args.layer_score,
        task_metrics=args.task_metrics,
        agent_events=args.agent_events,
        normalizers=args.normalizers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
