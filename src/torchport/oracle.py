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
from .parity import compare_tensor_archives
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
) -> dict[str, Any]:
    """执行一端，返回 RunManifest；manifest 在全部产物写完后原子提交。"""
    side = task[kind]
    run_dir = new_run_dir(root / "runs", trial, kind)
    job = {
        "kind": kind,
        "side": side,
        "inputs": task["inputs"],
        "seed": task["seed"],
        "mode": mode,
        "out_dir": str(run_dir),
        "weights": task["weights"] if kind == "source" else None,
        "oracle": str(oracle) if oracle else None,
        "performance": performance,
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
    if ok and (run_dir / "module_outputs.npz").is_file():
        artifacts["module_outputs"] = str(run_dir / "module_outputs.npz")
    for name in ("module_fp64", "module_alt"):
        if ok and (run_dir / f"{name}.npz").is_file():
            artifacts[name] = str(run_dir / f"{name}.npz")
    if ok and (run_dir / "module_inputs.npz").is_file():
        artifacts["module_inputs"] = str(run_dir / "module_inputs.npz")
    if ok and (run_dir / "module_calls.json").is_file():
        artifacts["module_calls"] = str(run_dir / "module_calls.json")
    if ok and (run_dir / "isolated.npz").is_file():
        artifacts["isolated"] = str(run_dir / "isolated.npz")
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


def verify(task: dict[str, Any], root: Path, *, trial: str = "t0") -> dict[str, Any]:
    """正式验收：冻结 oracle，candidate 正常运行 repeats 次，全部通过才算通过。"""
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
