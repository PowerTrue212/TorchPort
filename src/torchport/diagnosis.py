"""供受控 Agent 调用的公开检查、trace 和首次分歧定位。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .execution import add_physical_cost, read_json, run_physical_cost, write_json
from .oracle import run_side
from .parity import compare_noise_calibrated, compare_tensor_archives


def inspect_task(task: dict[str, Any]) -> dict[str, Any]:
    """返回公开任务契约，不读取或生成 oracle。"""
    def side(name: str) -> dict[str, Any]:
        value = task[name]
        return {
            "repo": value["repo"],
            "entry": value["entry"],
            "model": value["model"],
            "args": value["args"],
            "kwargs": value["kwargs"],
            "device": value["device"],
            "python": value.get("python"),
        }

    return {
        "kind": "torchport.inspect",
        "task_id": task["task_id"],
        "task_digest": task["digest"],
        "source": side("source"),
        "target": side("target"),
        "inputs": task["inputs"],
        "protocol": task["protocol"],
        "checkpoint": {"provided": bool(task.get("weights")), "sha256": task.get("weights_sha")},
        "candidate_identity": task["target_identity"],
    }


def trace(task: dict[str, Any], root: Path, *, trial: str = "t0") -> dict[str, Any]:
    """以同一输入和状态运行双端 trace，保存完整比较报告。"""
    root = Path(root).resolve()
    source = run_side(task, "source", root, trial=f"{trial}-trace-source", mode="trace")
    if not source["ok"]:
        report = _failed_trace(task, trial, "source", source)
    else:
        target = run_side(
            task,
            "target",
            root,
            trial=f"{trial}-trace-target",
            mode="trace",
            oracle=Path(source["dir"]),
        )
        if not target["ok"]:
            report = _failed_trace(task, trial, "target", target, source)
        elif "trace" not in source["artifacts"] or "trace" not in target["artifacts"]:
            report = {
                "kind": "torchport.trace",
                "task_id": task["task_id"],
                "task_digest": task["digest"],
                "trial_id": trial,
                "passed": False,
                "status": "evidence_missing",
                "failure_code": "TRACE_MISSING",
                "stage": "trace",
                "source_run": source,
                "target_run": target,
            }
        else:
            outputs = compare_tensor_archives(
                source["artifacts"]["outputs"],
                target["artifacts"]["outputs"],
                profile=task["protocol"]["profile"],
            )
            strict = compare_tensor_archives(
                source["artifacts"]["trace"],
                target["artifacts"]["trace"],
                profile="operator-strict",
            )
            isolated = None
            if source["artifacts"].get("module_outputs") and target["artifacts"].get("isolated"):
                isolated = compare_tensor_archives(
                    source["artifacts"]["module_outputs"],
                    target["artifacts"]["isolated"],
                    profile="operator-strict",
                    require_all=False,
                )
                isolated["capture"] = (target.get("capture") or {}).get("isolated")
                artifacts = source["artifacts"]
                if artifacts.get("module_fp64") and artifacts.get("module_alt"):
                    calls = json.loads(Path(artifacts["module_calls"]).read_text("utf-8"))
                    leaves = {call["module"] for call in calls if call.get("leaf")}
                    isolated["calibrated"] = compare_noise_calibrated(
                        artifacts["module_outputs"],
                        target["artifacts"]["isolated"],
                        artifacts["module_fp64"],
                        artifacts["module_alt"],
                        leaves=leaves,
                    )
            report = {
                "kind": "torchport.trace",
                "task_id": task["task_id"],
                "task_digest": task["digest"],
                "trial_id": trial,
                "passed": bool(outputs["passed"] and strict["passed"]),
                "status": "aligned" if outputs["passed"] and strict["passed"] else "diverged",
                "output": outputs,
                "strict": strict,
                "isolated": isolated,
                "source_run": source,
                "target_run": target,
            }
    report["physical_cost"] = add_physical_cost(
        run_physical_cost(report.get("source_run") or {}, task["source"]["device"]),
        run_physical_cost(report.get("target_run") or {}, task["target"]["device"]),
    )
    out = root / "trace" / f"{trial}.json"
    write_json(out, report)
    report["artifacts"] = {"report": str(out)}
    return report


def _failed_trace(
    task: dict[str, Any],
    trial: str,
    stage: str,
    run: dict[str, Any],
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "torchport.trace",
        "task_id": task["task_id"],
        "task_digest": task["digest"],
        "trial_id": trial,
        "passed": False,
        "status": "run_failed",
        "failure_code": "RUN_FAILED",
        "stage": stage,
        "source_run": source,
        "target_run": run if stage == "target" else None,
    }


def localize(report: dict[str, Any]) -> dict[str, Any]:
    """从已有 trace 报告提取固定的一处首次分歧；不重新访问 oracle。"""
    if report.get("status") in {"run_failed", "evidence_missing"}:
        return {
            "kind": "torchport.localization",
            "passed": False,
            "stage": report.get("stage"),
            "failure_code": report.get("failure_code"),
            "first_divergence": None,
        }
    strict = report.get("strict") or {}
    output = report.get("output") or {}
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


def load_and_localize(path: str | Path) -> dict[str, Any]:
    return localize(read_json(Path(path).resolve()))
