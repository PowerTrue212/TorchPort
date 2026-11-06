"""双端 trace、聚焦采集与模块隔离诊断。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..comparison import (
    ENVELOPE_V3,
    TOLERANCE_PROFILES,
    called_modules,
    compare_isolated_v3,
    compare_module_config,
    compare_noise_calibrated,
    compare_tensor_archives,
)
from ..runtime import add_physical_cost, read_json, run_physical_cost, write_json
from ..verification import build_oracle, run_side
from ..verification.noise import per_tensor_floors


def _trace_noise_floor(task: dict[str, Any], root: Path, trial: str) -> tuple[float, dict[str, float] | None]:
    """trace 用的 N_env：协议给定数值直接用（无逐算子值）；auto 时取同 trial 最近 verify 已标定的
    全局值与逐算子值（与候选无关）。"""
    value = task["protocol"].get("noise_floor", 0.0)
    if value != "auto":
        return float(value or 0.0), None
    path = Path(root) / "verify" / f"{trial}.json"
    noise = (read_json(path).get("noise") or {}) if path.is_file() else {}
    return float(noise.get("noise_floor") or 0.0), noise.get("per_op") or None


def _tolerance_profile(task: dict[str, Any]) -> str:
    """trace 的输出与重复性比较用固定容差；噪声包络协议（v2/v3）没有固定容差，退回 fp32-inference。"""
    profile = task["protocol"]["profile"]
    return profile if profile in TOLERANCE_PROFILES else "fp32-inference"


# trace 聚焦：首次分歧及其之前 FOCUS_WINDOW 个张量所属模块；磁盘余量低于下限时不启动。
FOCUS_WINDOW = 8
MIN_FREE_BYTES = 2 * 1024**3
# 每端聚焦 trace 写入上限；capture 在写 NPZ 前按数组大小估算。
MAX_TRACE_BYTES = 4 * 1024**3


def _refused(task: dict[str, Any], trial: str, code: str, stage: str, reason: str) -> dict[str, Any]:
    return {
        "kind": "torchport.trace",
        "task_id": task["task_id"],
        "task_digest": task["digest"],
        "trial_id": trial,
        "passed": False,
        "status": "run_failed",
        "failure_code": code,
        "stage": stage,
        "reason": reason,
        "source_run": None,
        "target_run": None,
    }


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

    只做聚焦采集：必须有同 trial 的 verify 报告且其中有首次分歧或临界位置，只记录其附近的模块。
    没有可聚焦的位置时不启动任何进程（全量逐层记录在 RAFT 上单次超过 20 GB）。
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
    if not focus:
        report = _refused(
            task, trial, "TRACE_NO_FOCUS", "preflight",
            "同一 trial 没有含首次分歧位置的 verify 报告；先运行 verify，且 verify 未通过时 trace 才有意义",
        )
        report["physical_cost"] = add_physical_cost()
        out = root / "trace" / f"{trial}.json"
        write_json(out, report)
        report["artifacts"] = {"report": str(out)}
        return report
    frozen = build_oracle(task, root)
    if frozen is not None and not frozen.get("ok"):
        report = {
            "kind": "torchport.trace",
            "task_id": task["task_id"],
            "task_digest": task["digest"],
            "trial_id": trial,
            "passed": False,
            "status": "run_failed",
            "failure_code": frozen.get("reason", "ORACLE_INVALID"),
            "stage": "oracle",
            "reason": "聚焦 trace 无法获取有效的冻结 oracle",
            "source_run": None,
            "target_run": None,
        }
        report["physical_cost"] = add_physical_cost(frozen.get("physical_cost"))
        out = root / "trace" / f"{trial}.json"
        write_json(out, report)
        report["artifacts"] = {"report": str(out)}
        return report
    frozen_dir = Path(frozen["dir"]) if frozen is not None else None
    source = run_side(
        task,
        "source",
        root,
        trial=f"{trial}-trace-source",
        mode="trace",
        oracle=frozen_dir,
        focus=focus,
        trace_max_bytes=MAX_TRACE_BYTES,
    )
    if not source["ok"]:
        report = _failed_trace(task, trial, "source", source)
    else:
        target = run_side(
            task,
            "target",
            root,
            trial=f"{trial}-trace-target",
            mode="trace",
            oracle=frozen_dir or Path(source["dir"]),
            module_oracle=Path(source["dir"]),
            focus=focus,
            trace_max_bytes=MAX_TRACE_BYTES,
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
                profile=_tolerance_profile(task),
            )
            strict = compare_tensor_archives(
                source["artifacts"]["trace"],
                target["artifacts"]["trace"],
                profile="operator-strict",
                # CountSketch 不保留逐元素容差语义；大条目的数值判定交给
                # v3 未压缩的 isolated 证据，这里仍严格核对原始 shape。
                skip_sketched=bool(focus),
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
                    profile = _tolerance_profile(task)
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
                    if task["protocol"]["profile"] == ENVELOPE_V3:
                        # v3：隔离比较按 max(D, N_env(算子)) 判定，localize 优先采用它
                        protocol = task["protocol"]
                        floor, per_op = _trace_noise_floor(task, root, trial)
                        floors = None
                        source_config = artifacts.get("module_config")
                        if per_op and source_config:
                            with np.load(artifacts["module_outputs"], allow_pickle=False) as archive:
                                names = list(archive.files)
                            floors = per_tensor_floors(names, read_json(Path(source_config)), per_op, floor)
                        isolated["envelope_v3"] = compare_isolated_v3(
                            artifacts["module_outputs"],
                            target["artifacts"]["isolated"],
                            artifacts["module_fp64"],
                            artifacts["module_alt"],
                            noise_floor=floor,
                            noise_floors=floors,
                            pass_k=float(protocol.get("pass_k", 2.0)),
                            floor_k=float(protocol.get("floor_k", 3.0)),
                            repeat_path=target["artifacts"].get("isolated_repeat"),
                        )
            configs = [run["artifacts"].get("module_config") for run in (source, target)]
            module_report = (
                compare_module_config(
                    *(read_json(Path(path)) for path in configs),
                    # 聚焦 trace 只记录少数模块，无法据此判断哪些模块被调用过；此时 used 留空
                    None if focus else called_modules(source["artifacts"]["trace"]),
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
        (frozen or {}).get("physical_cost"),
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
    contract = run.get("contract") or {}
    if contract.get("code") == "TRACE_TOO_LARGE":
        report = _refused(task, trial, "TRACE_TOO_LARGE", stage, contract.get("message", "trace 估算体积超限"))
        report["source_run"] = run if stage == "source" else source
        report["target_run"] = run if stage == "target" else None
        return report
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
