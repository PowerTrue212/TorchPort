"""Prepare, resume, and run every pre-registered trial in one experiment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from torchport.execution import build_env, write_json
from torchport.task import file_hash, load_task

from .aggregate import aggregate
from .deepseek import DeepSeekAgent, DeepSeekConfig, load_env
from .gateway import HarnessGateway
from .protocol import FeedbackLevel, TrialProtocol
from .runner import run_trial

AttemptRunner = Callable[[dict[str, Any], str, Path, Path, Path, str], dict[str, Any]]
CORE_RUNTIME_PACKAGES = ("torch", "jittor", "numpy", "scipy", "ftfy", "regex")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _canonical_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _inside(root: Path, value: str | Path) -> Path:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"实验路径越过 manifest 目录：{value}")
    return path


def _python_version(side: dict[str, Any], cache: Path, packages: tuple[str, ...]) -> dict[str, Any]:
    python = str(side["python"])
    script = f"""
import importlib.metadata as metadata
import json
import platform

versions = {{}}
for name in {packages!r}:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
print(json.dumps({{"python": platform.python_version(), "packages": versions}}, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [python, "-c", script],
            cwd=side["repo"],
            env=build_env(side, cache),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"python": python, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    text = completed.stdout.strip()
    details = None
    if completed.returncode == 0:
        try:
            details = json.loads(text.splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            details = None
    return {
        "python": python,
        "ok": completed.returncode == 0 and isinstance(details, dict),
        "exit_code": completed.returncode,
        "details": details,
        "stderr_tail": completed.stderr[-2000:],
    }


def _gpu_snapshot() -> dict[str, Any] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return {"command": command, "devices": completed.stdout.strip().splitlines()}


def preflight(manifest_path: str | Path) -> dict[str, Any]:
    """Validate the immutable experiment inputs without executing an Agent."""
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = _json(manifest_path)
    trials = manifest.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("experiment manifest 必须包含非空 trials")
    prompt = _inside(root, str(manifest.get("prompt") or ""))
    if not prompt.is_file():
        raise ValueError(f"experiment prompt 不存在：{prompt}")
    runtime_packages = manifest.get("runtime_packages") or []
    if not isinstance(runtime_packages, list) or any(
        not isinstance(name, str) or not name.strip() for name in runtime_packages
    ):
        raise ValueError("runtime_packages 必须是非空字符串列表")

    expected = {
        (task, condition, replicate)
        for task in manifest.get("tasks", [])
        for condition in manifest.get("conditions", [])
        for replicate in range(int(manifest.get("replicates", 1)))
    }
    observed: set[tuple[str, str, int]] = set()
    trial_ids: set[str] = set()
    output_paths: set[Path] = set()
    target_repos: set[Path] = set()
    task_records = []
    for item in trials:
        if not isinstance(item, dict):
            raise ValueError("trial 必须是对象")
        trial_id = str(item.get("trial_id") or "")
        if not trial_id or trial_id in trial_ids:
            raise ValueError(f"trial_id 缺失或重复：{trial_id!r}")
        trial_ids.add(trial_id)
        key = (str(item.get("task")), str(item.get("condition")), int(item.get("replicate", 0)))
        observed.add(key)
        public_path = _inside(root, str(item.get("public_task") or ""))
        hidden_path = _inside(root, str(item.get("hidden_task") or ""))
        output_path = _inside(root, str(item.get("path") or trial_id))
        if output_path in output_paths:
            raise ValueError(f"trial 输出路径重复：{output_path}")
        output_paths.add(output_path)
        public, hidden = load_task(public_path), load_task(hidden_path)
        public_target = Path(public["target"]["repo"]).resolve()
        hidden_target = Path(hidden["target"]["repo"]).resolve()
        if public_target != hidden_target:
            raise ValueError(f"{trial_id} 的 public/hidden target 不同")
        if public_target in target_repos:
            raise ValueError(f"不同 trial 共用了 candidate workspace：{public_target}")
        target_repos.add(public_target)
        task_records.append(
            {
                "trial_id": trial_id,
                "public_task_digest": public["digest"],
                "hidden_task_digest": hidden["digest"],
                "initial_candidate_identity": public["target_identity"],
                "source": public["source"],
                "target": public["target"],
            }
        )
    if expected and observed != expected:
        raise ValueError(
            f"trial 矩阵与 tasks/conditions/replicates 不一致：missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )

    artifacts = manifest.get("frozen_artifacts") or {}
    verified_artifacts = {}
    for relative, expected_hash in artifacts.items():
        path = _inside(root, relative)
        actual = file_hash(path)
        if actual != expected_hash:
            raise ValueError(f"冻结 artifact 摘要不匹配：{relative}")
        verified_artifacts[relative] = actual
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": file_hash(manifest_path),
        "prompt_sha256": file_hash(prompt),
        "task_records": task_records,
        "verified_artifacts": verified_artifacts,
    }


def _validate_isolation(
    manifest: dict[str, Any],
    attestation_path: Path | None,
    *,
    allow_trusted_pilot: bool,
) -> dict[str, Any] | None:
    isolation = manifest.get("isolation") or {}
    if not isolation.get("formal_claims_allowed"):
        if not allow_trusted_pilot:
            raise ValueError(
                "manifest 不允许正式结论；仅做可信 pilot 时显式传 --allow-trusted-pilot"
            )
        return None
    if attestation_path is None:
        raise ValueError("正式 experiment 必须提供 --worker-attestation")
    attestation = _json(attestation_path.resolve())
    required_true = (
        "network_isolation",
        "host_filesystem_isolation",
        "candidate_workspace_is_only_writable_mount",
        "source_and_frozen_inputs_read_only",
        "hidden_artifacts_not_mounted_for_agent",
    )
    missing = [name for name in required_true if attestation.get(name) is not True]
    if missing:
        raise ValueError(f"worker attestation 未满足隔离字段：{', '.join(missing)}")
    if not attestation.get("runtime") or not attestation.get("image_digest"):
        raise ValueError("worker attestation 必须记录 runtime 和 image_digest")
    return {**attestation, "sha256": file_hash(attestation_path)}


def _validate_frozen_plan(
    plan_path: Path,
    checked: dict[str, Any],
    *,
    allow_trusted_pilot: bool,
) -> None:
    plan = _json(plan_path)
    if plan.get("source_manifest_sha256") != checked["manifest_sha256"]:
        raise ValueError("experiment manifest 在 run-plan 冻结后发生变化")
    if plan.get("prompt_sha256") != checked["prompt_sha256"]:
        raise ValueError("experiment prompt 在 run-plan 冻结后发生变化")
    runtime_path = _inside(plan_path.parent, str(plan.get("trial_runtime_snapshot") or ""))
    if not runtime_path.is_file() or plan.get("trial_runtime_snapshot_sha256") != file_hash(
        runtime_path
    ):
        raise ValueError("runtime snapshot 在 run-plan 冻结后缺失或发生变化")
    runtime = _json(runtime_path)
    if runtime.get("tasks") != checked["task_records"]:
        raise ValueError("TaskSpec 或 candidate seed 在 run-plan 冻结后发生变化")
    isolation = checked["manifest"].get("isolation") or {}
    if not isolation.get("formal_claims_allowed"):
        if not allow_trusted_pilot:
            raise ValueError(
                "manifest 不允许正式结论；仅做可信 pilot 时显式传 --allow-trusted-pilot"
            )
        return
    attestation = runtime.get("worker_attestation") or {}
    if not attestation or plan.get("worker_attestation_sha256") != attestation.get("sha256"):
        raise ValueError("run-plan 缺少有效的冻结 worker attestation")


def prepare_run_plan(
    manifest_path: str | Path,
    *,
    worker_attestation: str | Path | None = None,
    allow_trusted_pilot: bool = False,
) -> Path:
    """Freeze preflight and runtime facts once; a resume reuses the same plan."""
    checked = preflight(manifest_path)
    root = checked["manifest_path"].parent
    plan_path = root / "run-plan.json"
    if plan_path.is_file():
        _validate_frozen_plan(
            plan_path,
            checked,
            allow_trusted_pilot=allow_trusted_pilot,
        )
        return plan_path

    attestation_path = Path(worker_attestation).resolve() if worker_attestation else None
    attestation = _validate_isolation(
        checked["manifest"], attestation_path, allow_trusted_pilot=allow_trusted_pilot
    )
    runtime_cache = root / ".preflight-cache"
    packages = tuple(
        dict.fromkeys(
            [*CORE_RUNTIME_PACKAGES, *(checked["manifest"].get("runtime_packages") or [])]
        )
    )
    runtimes: dict[str, Any] = {}
    for record in checked["task_records"]:
        for side_name in ("source", "target"):
            side = record[side_name]
            key = _canonical_hash(
                {
                    "python": side["python"],
                    "repo": side["repo"],
                    "device": side["device"],
                    "env": side.get("env") or {},
                    "packages": packages,
                }
            )[:16]
            if key not in runtimes:
                value = _python_version(side, runtime_cache / key, packages)
                if not value["ok"]:
                    raise ValueError(f"runtime preflight 失败：{value}")
                runtimes[key] = value
    needs_cuda = any(
        record[name]["device"] == "cuda"
        for record in checked["task_records"]
        for name in ("source", "target")
    )
    gpu = _gpu_snapshot()
    if needs_cuda and gpu is None:
        raise ValueError("TaskSpec 要求 CUDA，但 preflight 无法读取 nvidia-smi")
    runtime = {
        "schema_version": 1,
        "kind": "torchport.p2-runtime-snapshot",
        "captured_at": time.time(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "controller_python": sys.version,
        },
        "gpu": gpu,
        "runtimes": runtimes,
        "tasks": checked["task_records"],
        "worker_attestation": attestation,
    }
    runtime_path = root / "runtime-snapshot.json"
    write_json(runtime_path, runtime)
    plan = {
        **checked["manifest"],
        "kind": "torchport.p2-run-plan",
        "source_manifest": checked["manifest_path"].name,
        "source_manifest_sha256": checked["manifest_sha256"],
        "prompt_sha256": checked["prompt_sha256"],
        "trial_runtime_snapshot": runtime_path.name,
        "trial_runtime_snapshot_sha256": file_hash(runtime_path),
        "worker_attestation_sha256": attestation.get("sha256") if attestation else None,
    }
    write_json(plan_path, plan)
    return plan_path


def _prepare_attempt(
    root: Path, item: dict[str, Any], attempt_id: str, attempt_path: Path
) -> tuple[Path, Path]:
    public_seed_path = _inside(root, item["public_task"])
    public_seed = _json(public_seed_path)
    hidden_seed = _json(_inside(root, item["hidden_task"]))
    source_repo = Path(load_task(public_seed_path)["target"]["repo"]).resolve()
    attempt_root = attempt_path / "input"
    candidate = attempt_root / "repo"
    if attempt_root.exists():
        raise FileExistsError(f"attempt 输入已经存在：{attempt_root}")
    shutil.copytree(
        source_repo,
        candidate,
        ignore=shutil.ignore_patterns(".git", ".torchport", "__pycache__", "*.pyc"),
    )
    for value in (public_seed, hidden_seed):
        value["target"]["repo"] = str(candidate.resolve())
    public_path = attempt_root / "public.json"
    hidden_path = attempt_root / "hidden.json"
    write_json(public_path, public_seed)
    write_json(hidden_path, hidden_seed)
    write_json(
        attempt_root / "attempt.json",
        {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "planned_trial_id": item["trial_id"],
            "seed_candidate_identity": load_task(public_seed_path)["target_identity"],
            "candidate_repo": str(candidate.resolve()),
        },
    )
    return public_path, hidden_path


def execute_run_plan(
    plan_path: str | Path,
    run_attempt: AttemptRunner,
    *,
    max_infrastructure_retries: int = 1,
) -> dict[str, Any]:
    """Run or resume a plan. Every retry starts from a fresh copy of the frozen seed."""
    if max_infrastructure_retries < 0:
        raise ValueError("max_infrastructure_retries 不能为负数")
    plan_path = Path(plan_path).resolve()
    root = plan_path.parent
    plan = _json(plan_path)
    state_path = root / "run-state.json"
    state = (
        _json(state_path)
        if state_path.is_file()
        else {
            "schema_version": 1,
            "kind": "torchport.p2-run-state",
            "plan_sha256": file_hash(plan_path),
            "trials": {},
        }
    )
    if state.get("plan_sha256") != file_hash(plan_path):
        raise ValueError("run-plan 在实验开始后发生变化")

    for item in plan["trials"]:
        trial_id = str(item["trial_id"])
        record = state["trials"].setdefault(trial_id, {"attempts": [], "selected": None})
        for attempt in record["attempts"]:
            if attempt.get("status") == "running":
                attempt.update(status="infrastructure_error", error="controller_interrupted")
        valid = next((a for a in record["attempts"] if a.get("valid")), None)
        if valid:
            record["selected"] = valid["path"]
            continue
        while len(record["attempts"]) <= max_infrastructure_retries:
            number = len(record["attempts"])
            attempt_id = trial_id if number == 0 else f"{trial_id}-infra-r{number}"
            logical_path = _inside(root, item.get("path") or trial_id)
            attempt_path = logical_path / ("attempt-0" if number == 0 else f"infra-r{number}")
            attempt = {
                "attempt_id": attempt_id,
                "path": attempt_path.relative_to(root).as_posix(),
                "status": "running",
                "valid": False,
            }
            record["attempts"].append(attempt)
            write_json(state_path, state)
            try:
                public_task, hidden_task = _prepare_attempt(root, item, attempt_id, attempt_path)
                result = run_attempt(
                    item,
                    attempt_id,
                    public_task,
                    hidden_task,
                    attempt_path,
                    str(plan["trial_runtime_snapshot_sha256"]),
                )
                attempt.update(
                    status="complete",
                    valid=bool(result.get("valid")),
                    success=bool(result.get("success")),
                    stop_reason=result.get("stop_reason"),
                )
            except Exception as exc:
                attempt.update(
                    status="infrastructure_error",
                    valid=False,
                    success=False,
                    error_type=type(exc).__name__,
                    error=str(exc)[-2000:],
                )
            write_json(state_path, state)
            if attempt["valid"]:
                record["selected"] = attempt["path"]
                write_json(state_path, state)
                break
        if record["selected"] is None:
            record["exhausted"] = True
            write_json(state_path, state)
    return state


def _deepseek_runner(
    plan: dict[str, Any], root: Path, config: DeepSeekConfig, prompt: str
) -> AttemptRunner:
    budget = plan["budget"]

    def run(
        item: dict[str, Any],
        attempt_id: str,
        public_task: Path,
        hidden_task: Path,
        attempt_path: Path,
        runtime_sha256: str,
    ) -> dict[str, Any]:
        condition = FeedbackLevel(str(item["condition"]))
        gateway = HarnessGateway(
            public_task,
            root / ".evidence" / str(plan["experiment_id"]) / attempt_id,
            attempt_path,
            TrialProtocol(
                feedback=condition,
                max_oracle_queries=int(budget["oracle_queries"]),
                max_wall_seconds=float(budget["wall_seconds"]),
                allow_network_research=bool(plan.get("network_research", False)),
            ),
            hidden_task=hidden_task,
            trial_id=attempt_id,
            metadata={
                "experiment_id": plan["experiment_id"],
                "condition": condition.value,
                "replicate": int(item.get("replicate", 0)),
                "agent": {"provider": "deepseek", **config.public()},
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "token_budget": int(budget["total_tokens"]),
                "turn_budget": int(budget["model_turns"]),
                "trial_runtime_snapshot_sha256": runtime_sha256,
            },
        )
        agent = DeepSeekAgent(
            config,
            prompt,
            recorder=gateway.record_agent_event,
            budget=gateway.budget_status,
            max_total_tokens=int(budget["total_tokens"]),
        )
        return run_trial(gateway, agent)

    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--env", type=Path, default=Path("experiments/harness/.env"))
    parser.add_argument("--worker-attestation", type=Path)
    parser.add_argument("--allow-trusted-pilot", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-infrastructure-retries", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan_path = prepare_run_plan(
            args.manifest,
            worker_attestation=args.worker_attestation,
            allow_trusted_pilot=args.allow_trusted_pilot,
        )
        if args.prepare_only:
            print(json.dumps({"run_plan": str(plan_path)}, indent=2))
            return 0
        load_env(args.env)
        config = DeepSeekConfig.from_env()
        plan = _json(plan_path)
        if config.model != plan.get("model"):
            raise ValueError(
                f"DeepSeek model 与 run-plan 不一致：{config.model} != {plan.get('model')}"
            )
        config = DeepSeekConfig(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            request_timeout=config.request_timeout,
            max_tokens=config.max_tokens,
            temperature=float(plan.get("temperature", 0.0)),
            max_model_turns=int(plan["budget"]["model_turns"]),
            thinking=str(plan.get("thinking", "enabled")).lower() != "disabled",
            reasoning_effort=str(plan.get("reasoning_effort", "max")),
        )
        prompt_path = _inside(plan_path.parent, str(plan["prompt"]))
        prompt = prompt_path.read_text("utf-8")
        state = execute_run_plan(
            plan_path,
            _deepseek_runner(plan, plan_path.parent, config, prompt),
            max_infrastructure_retries=args.max_infrastructure_retries,
        )
        report = aggregate(plan_path)
        write_json(plan_path.parent / "summary.json", report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"experiment: {exc}") from exc
    print(json.dumps({"run_state": state, "summary": report}, ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
