"""只读聚合冻结的 P2 trial，不修改或挑选原始结果。"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)
PHYSICAL_FIELDS = ("executions", "execution_wall_seconds", "cuda_execution_seconds")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events = []
    for number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"事件必须是对象：{path}:{number}")
        events.append(value)
    return events


def _tokens(events: list[dict[str, Any]]) -> dict[str, int | None]:
    totals = {name: 0 for name in TOKEN_FIELDS}
    seen = {name: False for name in TOKEN_FIELDS}
    for event in events:
        if event.get("event") != "model_response":
            continue
        usage = event.get("usage") or {}
        for name in TOKEN_FIELDS:
            value = usage.get(name)
            if value is not None:
                totals[name] += int(value)
                seen[name] = True
    return {name: totals[name] if seen[name] else None for name in TOKEN_FIELDS}


def _add(target: dict[str, float | int], value: dict[str, Any], fields: tuple[str, ...]) -> None:
    for name in fields:
        target[name] += value.get(name) or 0


def _empty_physical() -> dict[str, float | int]:
    return {"executions": 0, "execution_wall_seconds": 0.0, "cuda_execution_seconds": 0.0}


def _tool_response(event: dict[str, Any], trial_dir: Path) -> dict[str, Any]:
    sequence = int(event.get("sequence") or 0)
    local = trial_dir / "tool-results" / f"{sequence:04d}.json"
    if local.is_file():
        return _json(local)
    artifact = event.get("response_artifact")
    if artifact and Path(artifact).is_file():
        return _json(Path(artifact))
    return {}


def _tool_metrics(events: list[dict[str, Any]], trial_dir: Path) -> dict[str, Any]:
    calls: Counter[str] = Counter()
    seconds: dict[str, float] = {}
    physical: dict[str, dict[str, float | int]] = {}
    previous = _empty_physical()
    candidates: set[str] = set()
    revisions = 0
    agent_started = next(
        (
            float(event["timestamp"])
            for event in events
            if event.get("event") == "agent_started" and event.get("timestamp") is not None
        ),
        None,
    )
    first_run_success = None
    first_verify_pass = None

    for event in events:
        if event.get("event") != "tool_call":
            continue
        tool = str(event.get("tool") or "unknown")
        calls[tool] += 1
        seconds[tool] = seconds.get(tool, 0.0) + float(event.get("duration_seconds") or 0.0)
        current = event.get("physical_evaluation_cost") or previous
        delta = {
            name: max(0, current.get(name, 0) - previous.get(name, 0)) for name in PHYSICAL_FIELDS
        }
        physical.setdefault(tool, _empty_physical())
        _add(physical[tool], delta, PHYSICAL_FIELDS)
        previous = {name: current.get(name, 0) for name in PHYSICAL_FIELDS}

        before, after = event.get("candidate_before"), event.get("candidate_after")
        candidates.update(str(value) for value in (before, after) if value)
        revisions += int(bool(before and after and before != after))

        response = _tool_response(event, trial_dir)
        data = response.get("data") or {}
        elapsed = None
        if agent_started is not None and event.get("timestamp") is not None:
            elapsed = max(0.0, float(event["timestamp"]) - agent_started)
        if tool == "run" and first_run_success is None and response.get("ok") and data.get("ok"):
            first_run_success = elapsed
        if (
            tool == "verify"
            and first_verify_pass is None
            and response.get("ok")
            and data.get("passed")
        ):
            first_verify_pass = elapsed

    return {
        "tool_calls_by_name": dict(sorted(calls.items())),
        "tool_time_seconds_by_name": dict(sorted(seconds.items())),
        "physical_evaluation_cost_by_tool": dict(sorted(physical.items())),
        "candidate_revisions": revisions,
        "unique_candidate_identities": len(candidates),
        "milestones_seconds": {
            "first_run_success": first_run_success,
            "first_development_verify_pass": first_verify_pass,
        },
    }


def aggregate(manifest_path: str | Path) -> dict[str, Any]:
    """按预注册 manifest 聚合；缺失和 invalid trial 保留，失败成本不填零。"""
    manifest_path = Path(manifest_path).resolve()
    manifest = _json(manifest_path)
    planned = manifest.get("trials")
    if not isinstance(planned, list) or not planned:
        raise ValueError("experiment manifest 必须包含非空 trials")
    state_path = manifest_path.parent / "run-state.json"
    run_state = _json(state_path) if state_path.is_file() else None
    state_trials = (run_state or {}).get("trials") or {}

    rows = []
    stop_reasons: Counter[str] = Counter()
    token_totals = {name: 0 for name in TOKEN_FIELDS}
    token_seen = {name: False for name in TOKEN_FIELDS}
    physical = _empty_physical()
    tool_calls: Counter[str] = Counter()
    tool_seconds: dict[str, float] = {}
    physical_by_tool: dict[str, dict[str, float | int]] = {}
    candidate_revisions = 0
    successful_times = []
    milestone_values: dict[str, list[float]] = {
        "first_run_success": [],
        "first_development_verify_pass": [],
        "official_verify_complete": [],
    }

    for item in planned:
        if not isinstance(item, dict) or not item.get("trial_id") or not item.get("condition"):
            raise ValueError("每个 trial 必须包含 trial_id 和 condition")
        relative = Path(item.get("path") or item["trial_id"])
        attempt_state = state_trials.get(str(item["trial_id"])) or {}
        selected = attempt_state.get("selected")
        attempts = list(attempt_state.get("attempts") or [])
        effective = selected or (attempts[-1].get("path") if attempts else None)
        trial_dir = (
            Path(effective)
            if effective and Path(effective).is_absolute()
            else manifest_path.parent / (effective or relative)
        )
        result_path = trial_dir / "result.json"
        exhausted = bool(attempt_state.get("exhausted"))
        completed_attempt = result_path.is_file()
        row: dict[str, Any] = {
            "trial_id": str(item["trial_id"]),
            "condition": str(item["condition"]),
            "replicate": int(item.get("replicate", 0)),
            "path": relative.as_posix() if not relative.is_absolute() else str(relative),
            "completed": completed_attempt or exhausted,
            "attempts": attempts,
            "selected_attempt": selected,
        }
        if not result_path.is_file():
            reason = "infrastructure_error" if exhausted else "not_run"
            row.update(valid=False, success=False, passed=None, stop_reason=reason)
            stop_reasons[reason] += 1
            rows.append(row)
            continue

        result = _json(result_path)
        events = _events(trial_dir / "events.jsonl")
        tokens = _tokens(events)
        tool_metrics = _tool_metrics(events, trial_dir)
        milestones = dict(tool_metrics["milestones_seconds"])
        milestones["official_verify_complete"] = result.get("total_wall_seconds")
        row.update(
            valid=bool(result.get("valid")),
            success=bool(result.get("success")),
            passed=bool(result.get("passed")),
            stop_reason=result.get("stop_reason"),
            logical_query_cost=result.get("logical_query_cost"),
            physical_evaluation_cost=result.get("physical_evaluation_cost"),
            agent_wall_seconds=result.get("agent_wall_seconds"),
            official_wall_seconds=result.get("official_wall_seconds"),
            total_wall_seconds=result.get("total_wall_seconds"),
            tokens=tokens,
            tool_calls_by_name=tool_metrics["tool_calls_by_name"],
            tool_time_seconds_by_name=tool_metrics["tool_time_seconds_by_name"],
            physical_evaluation_cost_by_tool=tool_metrics["physical_evaluation_cost_by_tool"],
            candidate_revisions=tool_metrics["candidate_revisions"],
            unique_candidate_identities=tool_metrics["unique_candidate_identities"],
            milestones_seconds=milestones,
        )
        stop_reasons[str(row["stop_reason"] or "unknown")] += 1
        if row["success"] and row["total_wall_seconds"] is not None:
            successful_times.append(float(row["total_wall_seconds"]))
        for name, value in tokens.items():
            if value is not None:
                token_totals[name] += value
                token_seen[name] = True
        total_physical = (result.get("physical_evaluation_cost") or {}).get("total") or {}
        _add(physical, total_physical, PHYSICAL_FIELDS)
        tool_calls.update(tool_metrics["tool_calls_by_name"])
        for name, value in tool_metrics["tool_time_seconds_by_name"].items():
            tool_seconds[name] = tool_seconds.get(name, 0.0) + value
        for name, value in tool_metrics["physical_evaluation_cost_by_tool"].items():
            physical_by_tool.setdefault(name, _empty_physical())
            _add(physical_by_tool[name], value, PHYSICAL_FIELDS)
        candidate_revisions += tool_metrics["candidate_revisions"]
        for name, value in milestones.items():
            if value is not None:
                milestone_values[name].append(float(value))
        rows.append(row)

    completed = sum(row["completed"] for row in rows)
    valid = sum(row["completed"] and row["valid"] for row in rows)
    successes = sum(row["completed"] and row["valid"] and row["success"] for row in rows)
    performance_path = manifest_path.parent / manifest.get("performance_path", "performance.json")
    performance = _json(performance_path) if performance_path.is_file() else None
    by_condition = {}
    for condition in sorted({row["condition"] for row in rows}):
        group = [row for row in rows if row["condition"] == condition]
        group_valid = sum(row["completed"] and row["valid"] for row in group)
        group_success = sum(row["completed"] and row["valid"] and row["success"] for row in group)
        by_condition[condition] = {
            "scheduled": len(group),
            "completed": sum(row["completed"] for row in group),
            "valid": group_valid,
            "successes": group_success,
            "success_rate": group_success / group_valid if group_valid else None,
        }

    return {
        "schema_version": 1,
        "kind": "torchport.p2-aggregate",
        "experiment_id": manifest.get("experiment_id"),
        "provider": manifest.get("provider"),
        "model": manifest.get("model"),
        "budget": manifest.get("budget"),
        "trial_runtime_snapshot_sha256": manifest.get("trial_runtime_snapshot_sha256"),
        "manifest": manifest_path.name,
        "complete": completed == len(rows),
        "scheduled_trials": len(rows),
        "completed_trials": completed,
        "valid_trials": valid,
        "successes": successes,
        "success_at_budget": {
            "successes": successes,
            "valid_trials": valid,
            "rate": successes / valid if valid else None,
        },
        "time_to_success_seconds": {
            "successful_trials": len(successful_times),
            "valid_trials": valid,
            "median": statistics.median(successful_times) if successful_times else None,
        },
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "tokens": {name: token_totals[name] if token_seen[name] else None for name in TOKEN_FIELDS},
        "cost": None,
        "physical_evaluation_cost": physical,
        "physical_evaluation_cost_by_tool": dict(sorted(physical_by_tool.items())),
        "tool_calls_by_name": dict(sorted(tool_calls.items())),
        "tool_time_seconds_by_name": dict(sorted(tool_seconds.items())),
        "candidate_revisions": candidate_revisions,
        "milestone_medians_seconds": {
            name: statistics.median(values) if values else None
            for name, values in milestone_values.items()
        },
        "performance_available": performance is not None,
        "performance": performance,
        "by_condition": by_condition,
        "trials": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate frozen TorchPort P2 trials")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = aggregate(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"aggregate: {exc}") from exc
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", "utf-8")
    print(text)
    return 0 if report["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
