"""供受控 Agent 调用的公开检查、trace 和首次分歧定位。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .execution import add_physical_cost, read_json, run_physical_cost, write_json
from .oracle import run_side
from .parity import (
    TOLERANCE_PROFILES,
    called_modules,
    compare_module_config,
    compare_noise_calibrated,
    compare_tensor_archives,
)


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


# trace 聚焦：首次分歧及其之前 FOCUS_WINDOW 个张量所属模块；磁盘余量低于下限时不启动。
FOCUS_WINDOW = 8
MIN_FREE_BYTES = 2 * 1024**3


def focus_modules(root: Path, trial: str, window: int = FOCUS_WINDOW) -> list[str] | None:
    """从同一 trial 最近一次 verify 的 E1 报告取首次分歧附近的模块名；没有可用报告时返回 None。"""
    path = Path(root) / "verify" / f"{trial}.json"
    if not path.is_file():
        return None
    e1 = read_json(path).get("e1") or {}
    first = e1.get("first_divergence") or e1.get("first_marginal")
    position = (first or {}).get("position")
    if not isinstance(position, int):
        return None
    modules: set[str] = set()
    for item in e1.get("tensors") or []:
        index = item.get("position")
        name = str(item.get("name") or "")
        if not isinstance(index, int) or not position - window <= index <= position or name.startswith("output"):
            continue
        base = name.split("#", 1)[0]
        modules.add(base)
        head, _, tail = base.rpartition(".")
        if head and tail.isdigit():
            modules.add(head)
    return sorted(modules) or None


def trace(task: dict[str, Any], root: Path, *, trial: str = "t0") -> dict[str, Any]:
    """以同一输入和状态运行双端 trace，保存完整比较报告。

    有同 trial 的 verify 报告时只记录首次分歧附近的模块（产物从 GB 级降到 MB 级）；否则全量记录。
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < MIN_FREE_BYTES:
        report = {
            "kind": "torchport.trace",
            "task_id": task["task_id"],
            "task_digest": task["digest"],
            "trial_id": trial,
            "passed": False,
            "status": "run_failed",
            "failure_code": "DISK_LOW",
            "stage": "preflight",
            "reason": f"证据目录可用空间 {free / 1024**3:.1f} GiB，低于 trace 下限 {MIN_FREE_BYTES / 1024**3:.0f} GiB",
            "source_run": None,
            "target_run": None,
            "physical_cost": add_physical_cost(),
        }
        out = root / "trace" / f"{trial}.json"
        write_json(out, report)
        report["artifacts"] = {"report": str(out)}
        return report
    focus = focus_modules(root, trial)
    source = run_side(task, "source", root, trial=f"{trial}-trace-source", mode="trace", focus=focus)
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
            focus=focus,
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
                if target["artifacts"].get("isolated_repeat"):
                    isolated["repeatability"] = compare_tensor_archives(
                        target["artifacts"]["isolated"],
                        target["artifacts"]["isolated_repeat"],
                        profile="operator-strict",
                    )
                    profile = task["protocol"]["profile"]
                    rtol, atol = TOLERANCE_PROFILES[profile]
                    verification_repeatability = compare_tensor_archives(
                        target["artifacts"]["isolated"],
                        target["artifacts"]["isolated_repeat"],
                        profile="operator-strict",
                        rtol=rtol,
                        atol=atol,
                    )
                    verification_repeatability["tolerance_basis"] = profile
                    isolated["repeatability_at_verification_tolerance"] = (
                        verification_repeatability
                    )
                artifacts = source["artifacts"]
                if artifacts.get("module_repeat"):
                    # source 自身重复性，与 target 的 repeatability 对称，只作诊断证据。
                    isolated["source_repeatability"] = compare_tensor_archives(
                        artifacts["module_outputs"],
                        artifacts["module_repeat"],
                        profile="operator-strict",
                        require_all=False,
                    )
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
            configs = [run["artifacts"].get("module_config") for run in (source, target)]
            module_report = (
                compare_module_config(
                    *(read_json(Path(path)) for path in configs),
                    called_modules(source["artifacts"]["trace"]),
                )
                if all(configs)
                else None
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
                # 同名模块构造超参差异（诊断证据，不参与判定）
                "module_config": module_report,
                # None 表示全量记录；否则只记录了这些模块
                "focus": focus,
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
    # 失败的那一端必须进入报告，并给出 stderr 最后一行；否则 Agent 只能看到 RUN_FAILED。
    execution = run.get("execution") or {}
    tail = (execution.get("stderr_tail") or "").strip()
    reason = "子进程超时" if execution.get("timeout") else (tail.splitlines() or ["无 stderr"])[-1]
    return {
        "kind": "torchport.trace",
        "task_id": task["task_id"],
        "task_digest": task["digest"],
        "trial_id": trial,
        "passed": False,
        "status": "run_failed",
        "failure_code": "RUN_FAILED",
        "stage": stage,
        "reason": f"{stage} 运行失败：{reason[-500:]}",
        "stderr_tail": tail,
        "source_run": run if stage == "source" else source,
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
            "reason": report.get("reason"),
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
