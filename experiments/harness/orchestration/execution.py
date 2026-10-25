"""按冻结 run plan 创建独立 attempt，并支持基础设施失败重试。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from torchport.contracts import file_hash, load_task
from torchport.runtime import write_json

from ..candidate import initialize_git_workspace
from ..protocol import INVALID_TRIAL_STOP_REASONS
from .common import inside as _inside
from .common import read_json as _json

AttemptRunner = Callable[[dict[str, Any], str, Path, Path, Path, str], dict[str, Any]]
TERMINAL_INVALID_REASONS = INVALID_TRIAL_STOP_REASONS - {"infrastructure_error"}


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
    initialize_git_workspace(candidate)
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
        if any(a.get("stop_reason") in TERMINAL_INVALID_REASONS for a in record["attempts"]):
            record["terminal_invalid"] = True
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
                    valid=bool(result.get("valid"))
                    and result.get("stop_reason") not in INVALID_TRIAL_STOP_REASONS,
                    success=bool(result.get("success"))
                    and result.get("stop_reason") not in INVALID_TRIAL_STOP_REASONS,
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
            if attempt.get("stop_reason") in TERMINAL_INVALID_REASONS:
                record["terminal_invalid"] = True
                break
        if record["selected"] is None and not record.get("terminal_invalid"):
            record["exhausted"] = True
        write_json(state_path, state)
    return state
