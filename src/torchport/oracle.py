"""Harness 编排：冻结 reference oracle → 运行 candidate → 比较 → 唯一判定。

run / evaluate / verify 三个动词都只组合这里的函数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .execution import (
    add_physical_cost,
    empty_physical_cost,
    execute,
    new_run_dir,
    read_json,
    run_physical_cost,
    write_json,
)
from .noise import calibrate as calibrate_noise
from .parity import (
    ENVELOPE_V3,
    called_modules,
    compare_envelope,
    compare_envelope_v3,
    compare_module_config,
    compare_tensor_archives,
)
from .task import file_hash
from .verify import judge, judge_repeats

ORACLE_KIND = "torchport.oracle"
MANIFEST = "manifest.json"


def _cache(root: Path, trial: str, side: str, python: str | None) -> Path:
    # 以 trial/side/环境 区分；同一 trial 内复用编译缓存，不同 trial 不共享可写缓存
    env = (python or "default").replace("\\", "/").replace("/", "_").replace(":", "")[-40:]
    return root / "cache" / trial / f"{side}-{env}"


def run_side(
    task: dict[str, Any],
    kind: str,
    root: Path,
    *,
    trial: str = "t0",
    mode: str = "normal",
    oracle: Path | None = None,
    performance: dict[str, int] | None = None,
    variant: dict[str, Any] | None = None,
    focus: list[str] | None = None,
) -> dict[str, Any]:
    """执行一端，返回 RunManifest；manifest 在全部产物写完后原子提交。

    focus 只用于 trace：只记录这些模块的输出与局部输入。
    """
    side = task[kind]
    run_dir = new_run_dir(root / "runs", trial, kind)
    job = {
        "focus": focus,
        # v3：layers 证据中的 output.* 不压缩，供逐元素比较
        "exact_outputs": task["protocol"]["profile"] == ENVELOPE_V3,
        "kind": kind,
        "side": side,
        "inputs": task["inputs"],
        "seed": task["seed"],
        "mode": mode,
        "out_dir": str(run_dir),
        "weights": task["weights"] if kind == "source" else None,
        "oracle": str(oracle) if oracle else None,
        "performance": performance,
        "variant": variant,
    }
    write_json(run_dir / "job.json", job)
    record = execute(
        side,
        ["_capture", str(run_dir / "job.json")],
        run_dir=run_dir,
        cache=_cache(root, trial, kind, side.get("python")),
        timeout=float(task["protocol"]["timeout"]),
    )
    capture = run_dir / "capture.json"
    # 路径存在不构成证据：只有本次进程成功且写出 capture.json 才算产物
    ok = record["ok"] and capture.is_file()
    artifacts = {"outputs": str(run_dir / "outputs.npz")} if ok else {}
    if ok and (run_dir / "trace.npz").is_file():
        artifacts["trace"] = str(run_dir / "trace.npz")
    if ok and (run_dir / "layers.npz").is_file():
        artifacts["layers"] = str(run_dir / "layers.npz")
    if ok and (run_dir / "module_outputs.npz").is_file():
        artifacts["module_outputs"] = str(run_dir / "module_outputs.npz")
    for name in ("module_fp64", "module_alt", "module_repeat"):
        if ok and (run_dir / f"{name}.npz").is_file():
            artifacts[name] = str(run_dir / f"{name}.npz")
    if ok and (run_dir / "module_inputs.npz").is_file():
        artifacts["module_inputs"] = str(run_dir / "module_inputs.npz")
    if ok and (run_dir / "module_config.json").is_file():
        artifacts["module_config"] = str(run_dir / "module_config.json")
    if ok and (run_dir / "module_shapes.json").is_file():
        artifacts["module_shapes"] = str(run_dir / "module_shapes.json")
    if ok and (run_dir / "module_calls.json").is_file():
        artifacts["module_calls"] = str(run_dir / "module_calls.json")
    if ok and (run_dir / "isolated.npz").is_file():
        artifacts["isolated"] = str(run_dir / "isolated.npz")
    if ok and (run_dir / "isolated_repeat.npz").is_file():
        artifacts["isolated_repeat"] = str(run_dir / "isolated_repeat.npz")
    manifest = {
        "kind": "torchport.run",
        "run_id": run_dir.name,
        "trial_id": trial,
        "side": kind,
        "mode": mode,
        "task_id": task["task_id"],
        "task_digest": task["digest"],
        # TaskSpec 已冻结本端内容身份；repeat 不再重复遍历整个仓库。
        "side_identity": task[f"{kind}_identity"],
        "oracle": str(oracle) if oracle else None,
        "execution": record,
        "ok": ok,
        "dir": str(run_dir),
        "capture": read_json(capture) if ok else None,
        "artifacts": artifacts,
    }
    write_json(run_dir / MANIFEST, manifest)
    return manifest


def build_oracle(task: dict[str, Any], root: Path, *, rebuild: bool = False) -> dict[str, Any]:
    """source 运行 repeats 次，输出须逐位稳定，才冻结为 oracle；协议摘要不变时复用。"""
    path = root / "oracle" / task["digest"][:16]
    meta_path = path / "oracle.json"
    if meta_path.is_file() and not rebuild:
        meta = read_json(meta_path)
        if _oracle_valid(meta, task, path):
            return {
                **meta,
                "oracle_reused": True,
                "physical_cost": empty_physical_cost(),
            }
    runs = [
        run_side(task, "source", root, trial=f"oracle-{i}")
        for i in range(int(task["protocol"]["repeats"]))
    ]
    failed = next((run for run in runs if not run["ok"]), None)
    physical = add_physical_cost(
        *(run_physical_cost(run, task["source"]["device"]) for run in runs)
    )
    if failed:
        return {
            "kind": ORACLE_KIND,
            "ok": False,
            "reason": "SOURCE_RUN_FAILED",
            "run": failed,
            "physical_cost": physical,
        }
    hashes = {run["capture"]["hashes"]["output"] for run in runs}
    if len(hashes) != 1:
        return {
            "kind": ORACLE_KIND,
            "ok": False,
            "reason": "ORACLE_UNSTABLE",
            "runs": runs,
            "physical_cost": physical,
        }
    first = Path(runs[0]["dir"])
    path.mkdir(parents=True, exist_ok=True)
    for name in ("inputs.npz", "outputs.npz", "state.npz"):
        (path / name).write_bytes((first / name).read_bytes())
    meta = {
        "kind": ORACLE_KIND,
        "ok": True,
        "task_id": task["task_id"],
        "task_digest": task["digest"],
        "dir": str(path),
        "capture": runs[0]["capture"],
        "files": {name: file_hash(path / name) for name in ("inputs.npz", "outputs.npz", "state.npz")},
        "source_runs": [run["run_id"] for run in runs],
        "build_physical_cost": physical,
    }
    write_json(meta_path, meta)
    return {**meta, "oracle_reused": False, "physical_cost": physical}


def load_oracle(task: dict[str, Any], root: Path) -> dict[str, Any]:
    """只读加载已冻结的 oracle；缺失或身份不符时返回不可用，不启动模型。"""
    path = root / "oracle" / task["digest"][:16]
    meta_path = path / "oracle.json"
    if not meta_path.is_file():
        return {"kind": ORACLE_KIND, "ok": False, "reason": "ORACLE_MISSING"}
    meta = read_json(meta_path)
    if not _oracle_valid(meta, task, path):
        return {"kind": ORACLE_KIND, "ok": False, "reason": "ORACLE_INVALID"}
    return {**meta, "oracle_reused": True, "physical_cost": empty_physical_cost()}


def _oracle_valid(meta: dict[str, Any], task: dict[str, Any], path: Path) -> bool:
    # 按内容与协议身份校验，不按目录名或时间戳猜测
    if not meta.get("ok") or meta.get("task_digest") != task["digest"]:
        return False
    files = meta.get("files") or {}
    return all((path / n).is_file() and file_hash(path / n) == h for n, h in files.items()) and bool(
        files
    )


def evaluate(
    oracle: dict[str, Any],
    run: dict[str, Any],
    protocol: dict[str, Any],
    candidate: str | None = None,
) -> dict[str, Any]:
    """只读既有产物并比较，不启动模型；身份不符直接拒绝。"""
    if not oracle.get("ok"):
        result = judge(None, None, None, None, protocol)
        result.update(failure_code=oracle.get("reason", "ORACLE_INVALID"))
        result["summary"] = f"{result['failure_code']}: oracle 不可用"
        return result
    if run.get("task_digest") != oracle["task_digest"] or run.get("oracle") != oracle["dir"]:
        raise ValueError("候选产物与 oracle 的 task/协议身份不符")
    if run.get("side") != "target" or not run.get("side_identity"):
        raise ValueError("候选产物缺少 target 端内容身份")
    if candidate is not None and run["side_identity"] != candidate:
        raise ValueError("候选产物与当前 target 代码/配置不符，需重新 run")
    if not run.get("ok"):
        execution = run.get("execution") or {}
        result = judge(oracle["capture"], None, None, None, protocol)
        tail = (execution.get("stderr_tail") or "").strip()
        reason = "子进程超时" if execution.get("timeout") else (tail.splitlines() or ["无 stderr"])[-1]
        # 失败原因必须进入结果，Agent 不必翻日志
        result["problems"].append(f"target 运行失败：{reason[-500:]}")
        result["summary"] = f"{result['failure_code']}: {'; '.join(result['problems'])}"
        result["stderr_tail"] = tail
        result["timeout"] = execution.get("timeout")
        result["run_id"] = run.get("run_id")
        return result
    reference = Path(oracle["dir"]) / "outputs.npz"
    candidate_path = Path(run["artifacts"]["outputs"])
    fp32 = compare_tensor_archives(reference, candidate_path, profile=protocol["profile"])
    strict = compare_tensor_archives(reference, candidate_path, profile="operator-strict")
    result = judge(oracle["capture"], run["capture"], fp32, strict, protocol)
    result["run_id"] = run["run_id"]
    result["candidate_identity"] = run["side_identity"]
    result["artifacts"] = {"reference": str(reference), "candidate": str(candidate_path)}
    return result


def build_envelope(task: dict[str, Any], root: Path, oracle: dict[str, Any]) -> dict[str, Any]:
    """在 oracle 状态与输入上采集 source 整网逐层激活：reference 一份 + 每个合法变体一份。

    变体集合属于 protocol，已进入 task digest；oracle 目录变化即重建，不跨协议复用。
    """
    path = Path(oracle["dir"]) / "envelope"
    meta_path = path / "envelope.json"
    variants = list(task["protocol"].get("variants") or [])
    if meta_path.is_file():
        meta = read_json(meta_path)
        files = [meta.get("reference"), *(meta.get("variants") or [])]
        # v3 自动标定需要参考运行的模块输入形状；旧 envelope 缺这份元数据时重建，不静默退回默认形状
        needs_shapes = task["protocol"].get("noise_floor") == "auto"
        shapes_ok = not needs_shapes or (meta.get("module_shapes") and Path(meta["module_shapes"]).is_file())
        if meta.get("task_digest") == task["digest"] and all(f and Path(f).is_file() for f in files) and shapes_ok:
            return {**meta, "physical_cost": empty_physical_cost()}
    runs = [run_side(task, "source", root, trial="envelope-ref", mode="layers")]
    for index, variant in enumerate(variants):
        runs.append(
            run_side(task, "source", root, trial=f"envelope-v{index}", mode="layers", variant=variant)
        )
    physical = add_physical_cost(*(run_physical_cost(run, task["source"]["device"]) for run in runs))
    failed = next((run for run in runs if not run["ok"] or "layers" not in run["artifacts"]), None)
    if failed:
        return {"ok": False, "reason": "SOURCE_RUN_FAILED", "run": failed, "physical_cost": physical}
    path.mkdir(parents=True, exist_ok=True)
    meta = {
        "ok": True,
        "task_digest": task["digest"],
        "reference": runs[0]["artifacts"]["layers"],
        "variants": [run["artifacts"]["layers"] for run in runs[1:]],
        "applied": [run["capture"].get("variant") for run in runs[1:]],
        # 参考运行的模块构造超参，供 verify 与 target 同名模块逐键比较（诊断证据）
        "module_config": runs[0]["artifacts"].get("module_config"),
        # 参考运行的模块输入形状：N_env 按本任务真实的全局归约规模标定
        "module_shapes": runs[0]["artifacts"].get("module_shapes"),
        "build_physical_cost": physical,
    }
    write_json(meta_path, meta)
    return {**meta, "physical_cost": physical}


def verify_envelope(task: dict[str, Any], root: Path, *, trial: str = "t0") -> dict[str, Any]:
    """协议 v2（noise-envelope）：target 整网逐层激活落在 source 合法变体包络 + target 自身波动之内。"""
    root = Path(root).resolve()
    protocol = task["protocol"]
    oracle = build_oracle(task, root)
    envelope = build_envelope(task, root, oracle) if oracle.get("ok") else {"ok": False}
    target_runs: list[dict[str, Any]] = []
    result: dict[str, Any]
    report_e1 = None
    module_report = None
    # noise_floor="auto"：按 target 环境实测 N_env（与候选无关，按环境缓存）；数值则直接使用
    noise: dict[str, Any] = {"physical_cost": empty_physical_cost()}
    noise_floor = 0.0
    if protocol["profile"] == ENVELOPE_V3 and protocol.get("noise_floor") == "auto":
        noise = calibrate_noise(task, root, envelope) if oracle.get("ok") and envelope.get("ok") else noise
        noise_floor = float(noise.get("noise_floor") or 0.0)
    elif protocol["profile"] == ENVELOPE_V3:
        noise_floor = float(protocol.get("noise_floor", 0.0))
    if not oracle.get("ok"):
        result = evaluate(oracle, {}, protocol)
    elif not envelope.get("ok"):
        result = judge(oracle["capture"], None, None, None, protocol)
        result.update(failure_code="SOURCE_RUN_FAILED", summary="SOURCE_RUN_FAILED: source 合法变体运行失败")
    elif noise.get("ok") is False:
        # 标定失败不能默认 N_env=0 继续判定，也不能默认放宽；直接失败并给出原因
        result = judge(oracle["capture"], None, None, None, protocol)
        tail = (noise.get("stderr_tail") or "").strip().splitlines() or ["无 stderr"]
        result.update(
            failure_code="NOISE_PROBE_FAILED",
            summary=f"NOISE_PROBE_FAILED: target 环境噪声标定失败：{tail[-1][-500:]}",
        )
    else:
        for _ in range(max(int(protocol["repeats"]), 2)):
            target_runs.append(
                run_side(task, "target", root, trial=trial, mode="layers", oracle=Path(oracle["dir"]))
            )
        first = target_runs[0]
        if not all(run["ok"] and "layers" in run["artifacts"] for run in target_runs):
            bad = next(run for run in target_runs if not run["ok"] or "layers" not in run["artifacts"])
            result = evaluate(oracle, {**bad, "ok": False}, protocol, task["target_identity"])
        else:
            if protocol["profile"] == ENVELOPE_V3:
                variants = list(protocol.get("variants") or [])
                report_e1 = compare_envelope_v3(
                    envelope["reference"],
                    envelope["variants"],
                    [run["artifacts"]["layers"] for run in target_runs],
                    pass_k=float(protocol.get("pass_k", 2.0)),
                    floor_k=float(protocol.get("floor_k", 3.0)),
                    noise_floor=noise_floor,
                    batch_variants=tuple(
                        i for i, v in enumerate(variants) if int(v.get("batch", 1)) > 1
                    ),
                )
            else:
                report_e1 = compare_envelope(
                    envelope["reference"],
                    envelope["variants"],
                    first["artifacts"]["layers"],
                    [run["artifacts"]["layers"] for run in target_runs[1:]],
                )
            structure = compare_tensor_archives(
                Path(oracle["dir"]) / "outputs.npz", first["artifacts"]["outputs"], profile="fp32-inference"
            )
            result = judge(oracle["capture"], first["capture"], structure, None, protocol, numerical=report_e1)
            result["run_id"] = first["run_id"]
            result["candidate_identity"] = first["side_identity"]
            source_config = envelope.get("module_config")
            target_config = first["artifacts"].get("module_config")
            if source_config and target_config and Path(source_config).is_file():
                # 诊断证据：同名模块构造超参（eps、padding、align_corners 等）逐键比较，不参与判定
                module_report = compare_module_config(
                    read_json(Path(source_config)),
                    read_json(Path(target_config)),
                    called_modules(envelope["reference"]),
                )
    target_physical = add_physical_cost(
        *(run_physical_cost(run, task["target"]["device"]) for run in target_runs)
    )
    report = {
        "kind": "torchport.verify",
        "protocol": protocol["profile"],
        # v3 三档：pass / marginal / fail；v2 只有 passed
        "e1_status": (report_e1 or {}).get("status"),
        "task_id": task["task_id"],
        "task_digest": task["digest"],
        "trial_id": trial,
        "passed": bool(result.get("passed")),
        "status": "verified" if result.get("passed") else "failed",
        "failure_code": result.get("failure_code"),
        "summary": result.get("summary"),
        "next": result.get("next"),
        "e1": report_e1,
        "module_config": module_report,
        "physical_cost": add_physical_cost(
            oracle.get("physical_cost"), envelope.get("physical_cost"), noise.get("physical_cost"), target_physical
        ),
        "noise": {k: noise.get(k) for k in ("noise_floor", "dominant_op", "per_op", "jittor", "gpu", "reused")}
        if noise.get("ok")
        else None,
        "oracle": oracle.get("dir"),
        "envelope": {k: envelope.get(k) for k in ("reference", "variants", "applied")},
        "results": [result],
    }
    out = root / "verify" / f"{trial}.json"
    write_json(out, report)
    report["artifacts"] = {"report": str(out)}
    return report


def verify(task: dict[str, Any], root: Path, *, trial: str = "t0") -> dict[str, Any]:
    """正式验收：冻结 oracle，candidate 正常运行 repeats 次，全部通过才算通过。"""
    if task["protocol"]["profile"] in ("noise-envelope", ENVELOPE_V3):
        return verify_envelope(task, root, trial=trial)
    root = Path(root).resolve()
    oracle = build_oracle(task, root)
    protocol = task["protocol"]
    results = []
    target_runs = []
    for _ in range(int(protocol["repeats"])):
        if not oracle.get("ok"):
            results.append(evaluate(oracle, {}, protocol))
            break
        run = run_side(task, "target", root, trial=trial, oracle=Path(oracle["dir"]))
        target_runs.append(run)
        results.append(evaluate(oracle, run, protocol, task["target_identity"]))
        if not results[-1]["passed"]:
            break  # 一次失败即不可能全部通过，保留已有证据即可
    repeat = judge_repeats(results, int(protocol["repeats"]))
    target_physical = add_physical_cost(
        *(
            run_physical_cost(run, task["target"]["device"])
            for run in target_runs
        )
    )
    report = {
        "kind": "torchport.verify",
        "task_id": task["task_id"],
        "task_digest": task["digest"],
        "trial_id": trial,
        "passed": repeat["passed"],
        "status": "verified" if repeat["passed"] else "failed",
        "failure_code": repeat["failure_code"],
        "aligned": bool(results) and all(result.get("aligned") is True for result in results),
        "strict_aligned": bool(results)
        and all(result.get("strict_aligned") is True for result in results),
        "mismatch": results[-1].get("mismatch") if results else None,
        "physical_cost": add_physical_cost(oracle.get("physical_cost"), target_physical),
        "repeats": repeat,
        "summary": results[-1]["summary"] if results else "未运行",
        "next": results[-1].get("next") if results else None,
        "oracle": oracle.get("dir"),
        "results": results,
    }
    out = root / "verify" / f"{trial}.json"
    write_json(out, report)
    report["artifacts"] = {"report": str(out)}
    return report
