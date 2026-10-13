"""成熟代码 Agent 的 TorchPort 反馈、预算与最终认证边界。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from torchport.contracts import file_hash, load_task
from torchport.diagnosis import inspect_task, localize, trace
from torchport.runtime import add_physical_cost, empty_physical_cost, run_physical_cost, write_json
from torchport.verification import build_oracle, evaluate, load_oracle, run_side, verify

from .candidate import finalize_candidate
from .feedback import filter_feedback, public_run, trace_summary
from .protocol import TrialProtocol
from .tooling import ORACLE_TOOLS, event_arguments, tool_schemas, validate_call


class HarnessGateway:
    """一次 trial 的有状态 TorchPort 接口；不接管 Agent 的通用代码工具。"""

    def __init__(
        self,
        task: str | Path,
        evidence_root: str | Path,
        trial_dir: str | Path,
        protocol: TrialProtocol,
        *,
        hidden_task: str | Path | None = None,
        trial_id: str = "t0",
        metadata: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.task_path = Path(task).resolve()
        self.hidden_task_path = Path(hidden_task).resolve() if hidden_task else self.task_path
        self.evidence_root = Path(evidence_root).resolve()
        self.trial_dir = Path(trial_dir).resolve()
        self.protocol = protocol
        self.trial_id = trial_id
        self.clock = clock
        self.started = clock()
        self.oracle_queries = 0
        self._queried_candidates: set[str] = set()
        self._queries_by_tool = {name: 0 for name in sorted(ORACLE_TOOLS)}
        self._physical_cost = empty_physical_cost()
        self.calls = 0
        self._runs: dict[str, dict[str, Any]] = {}
        self._last_run_id: str | None = None
        self._last_trace: dict[str, Any] | None = None

        task_value = self._task()
        self.workspace = Path(task_value["target"]["repo"]).resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"candidate 工作区不存在：{self.workspace}")
        hidden_target = Path(load_task(self.hidden_task_path)["target"]["repo"]).resolve()
        if hidden_target != self.workspace:
            raise ValueError("开发 TaskSpec 与隐藏 TaskSpec 必须指向同一 candidate 工作区")
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.trial_dir / "events.jsonl"
        trial = {
            "schema_version": 1,
            "created_at": time.time(),
            "trial_id": trial_id,
            "task": str(self.task_path),
            "hidden_task": str(self.hidden_task_path),
            "task_id": task_value["task_id"],
            "task_digest": task_value["digest"],
            "workspace": str(self.workspace),
            "initial_candidate_identity": task_value["target_identity"],
            "protocol": protocol.as_dict(),
        }
        for key, value in (metadata or {}).items():
            if key in trial:
                raise ValueError(f"trial metadata 不能覆盖保留字段：{key}")
            trial[key] = value
        write_json(self.trial_dir / "trial.json", trial)

    def tool_schemas(self) -> list[dict[str, Any]]:
        return tool_schemas(self.protocol)

    def call(self, tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """调用一个白名单工具，并追加不可变事件。"""
        arguments = dict(arguments or {})
        started = self.clock()
        before = self._candidate_identity()
        response: dict[str, Any]
        self.calls += 1
        try:
            self._check_wall_budget()
            self._validate_call(tool, arguments)
            if tool in ORACLE_TOOLS:
                self._spend_oracle_query(tool)
            response = {"ok": True, "data": self._dispatch(tool, arguments)}
        except Exception as exc:  # 工具边界必须返回结构化错误，不能击穿 Agent loop
            response = {
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        response["budget"] = self.budget_status()
        finished = self.clock()
        artifact = self.trial_dir / "tool-results" / f"{self.calls:04d}.json"
        write_json(artifact, response)
        self._append_event(
            {
                "schema_version": 1,
                "sequence": self.calls,
                "event": "tool_call",
                "timestamp": time.time(),
                "tool": tool,
                "arguments": event_arguments(tool, arguments),
                "started": started,
                "finished": finished,
                "duration_seconds": max(0.0, finished - started),
                "candidate_before": before,
                "candidate_after": self._candidate_identity(),
                "oracle_queries": self.oracle_queries,
                "logical_query_cost": dict(self._queries_by_tool),
                "physical_evaluation_cost": dict(self._physical_cost),
                "response_ok": response["ok"],
                "error_type": (response.get("error") or {}).get("type"),
                "response_artifact": str(artifact),
                "response_sha256": file_hash(artifact),
            }
        )
        return response

    def record_agent_event(self, event: str, **facts: Any) -> None:
        """供可信 supervisor 记录检查、usage、turn 和停止事件；不暴露为 Agent tool。"""
        self._append_event(
            {
                "schema_version": 1,
                "event": event,
                "timestamp": time.time(),
                **facts,
            }
        )

    def budget_status(self) -> dict[str, Any]:
        elapsed = max(0.0, self.clock() - self.started)
        query_remaining = max(0, self.protocol.max_oracle_queries - self.oracle_queries)
        wall_remaining = max(0.0, self.protocol.max_wall_seconds - elapsed)
        return {
            "logical_query_cost": {
                "total": self.oracle_queries,
                "by_tool": dict(self._queries_by_tool),
                "unique_candidates": len(self._queried_candidates),
            },
            "physical_evaluation_cost": dict(self._physical_cost),
            "oracle_queries": self.oracle_queries,
            "unique_candidate_queries": len(self._queried_candidates),
            "max_oracle_queries": self.protocol.max_oracle_queries,
            "wall_seconds": elapsed,
            "max_wall_seconds": self.protocol.max_wall_seconds,
            "oracle_queries_remaining": query_remaining,
            "wall_seconds_remaining": wall_remaining,
            "budget_stop_reason": (
                "wall_budget_exhausted"
                if wall_remaining <= 0
                else "oracle_query_budget_exhausted" if query_remaining <= 0 else None
            ),
        }

    def certify(self, stop_reason: str) -> dict[str, Any]:
        """trial controller 专用；不出现在 tool_schemas，也不能通过 call 调用。"""
        official_started = self.clock()
        agent_wall = max(0.0, official_started - self.started)
        delivery = finalize_candidate(
            self.workspace,
            self.trial_dir / "delivery",
            required_entry=self.protocol.candidate_entry,
        )
        official_task_value = load_task(self.hidden_task_path)
        official_task_value["target"]["repo"] = delivery["snapshot"]
        official_task_path = self.trial_dir / "delivery" / "official-task.json"
        write_json(official_task_path, official_task_value)
        official_task = load_task(official_task_path)
        report = verify(
            official_task,
            self.evidence_root / "official",
            trial=f"{self.trial_id}-official",
        )
        official_wall = max(0.0, self.clock() - official_started)
        write_json(self.trial_dir / "official-verify.json", report)
        official_physical = report.get("physical_cost") or empty_physical_cost()
        total_physical = add_physical_cost(self._physical_cost, official_physical)
        result = {
            "schema_version": 1,
            "trial_id": self.trial_id,
            "passed": bool(report["passed"]),
            "valid": stop_reason != "infrastructure_error",
            "stop_reason": stop_reason,
            "feedback": self.protocol.feedback.value,
            "oracle_queries": self.oracle_queries,
            "unique_candidate_queries": len(self._queried_candidates),
            "logical_query_cost": {
                "total": self.oracle_queries,
                "by_tool": dict(self._queries_by_tool),
                "unique_candidates": len(self._queried_candidates),
            },
            "physical_evaluation_cost": {
                "agent": dict(self._physical_cost),
                "official": dict(official_physical),
                "total": total_physical,
            },
            "official_wall_seconds": official_wall,
            "tool_calls": self.calls,
            "agent_wall_seconds": agent_wall,
            "candidate_identity": official_task["target_identity"],
            "candidate_delivery": delivery,
            "official_report": str(self.trial_dir / "official-verify.json"),
        }
        result["total_wall_seconds"] = result["agent_wall_seconds"] + result[
            "official_wall_seconds"
        ]
        result["success"] = result["valid"] and result["passed"]
        write_json(self.trial_dir / "result.json", result)
        self._append_event({"schema_version": 1, "event": "official_verify", **result})
        return result

    def _dispatch(self, tool: str, arguments: dict[str, Any]) -> Any:
        if tool == "inspect":
            return inspect_task(self._task())
        if tool == "run":
            task = self._task()
            oracle = build_oracle(task, self.evidence_root)
            if not oracle.get("ok"):
                self._add_physical(oracle.get("physical_cost"))
                return self._feedback(oracle)
            manifest = run_side(
                task,
                "target",
                self.evidence_root,
                trial=self.trial_id,
                oracle=Path(oracle["dir"]),
            )
            self._runs[manifest["run_id"]] = manifest
            self._last_run_id = manifest["run_id"]
            cost = add_physical_cost(
                oracle.get("physical_cost"),
                run_physical_cost(manifest, task["target"]["device"]),
            )
            self._add_physical(cost)
            return {**public_run(manifest), "physical_cost": cost}
        if tool == "evaluate":
            run_id = str(arguments.get("run_id") or self._last_run_id or "")
            if run_id not in self._runs:
                raise ValueError("没有可评估的 run_id；请先调用 run")
            task = self._task()
            report = evaluate(
                load_oracle(task, self.evidence_root),
                self._runs[run_id],
                task["protocol"],
                task["target_identity"],
            )
            return self._feedback(report)
        if tool == "verify":
            report = verify(self._task(), self.evidence_root, trial=self.trial_id)
            self._add_physical(report.get("physical_cost"))
            return self._feedback(report)
        if tool == "trace":
            self._last_trace = trace(self._task(), self.evidence_root, trial=self.trial_id)
            self._add_physical(self._last_trace.get("physical_cost"))
            return trace_summary(self._last_trace)
        if tool == "localize":
            if self._last_trace is None:
                raise ValueError("没有 trace；请先调用 trace")
            return localize(self._last_trace)
        raise ValueError(f"未知或未授权工具：{tool}")

    def _validate_call(self, tool: str, arguments: dict[str, Any]) -> None:
        validate_call(self.protocol, tool, arguments)

    def _feedback(self, report: dict[str, Any]) -> dict[str, Any]:
        return filter_feedback(report, self.protocol.feedback)

    def _task(self) -> dict[str, Any]:
        return load_task(self.task_path)

    def _candidate_identity(self) -> str:
        return self._task()["target_identity"]

    def _check_wall_budget(self) -> None:
        if self.clock() - self.started >= self.protocol.max_wall_seconds:
            raise TimeoutError("wall budget 已耗尽")

    def _spend_oracle_query(self, tool: str) -> None:
        if self.oracle_queries >= self.protocol.max_oracle_queries:
            raise RuntimeError("oracle query budget 已耗尽")
        self.oracle_queries += 1
        self._queries_by_tool[tool] += 1
        self._queried_candidates.add(self._candidate_identity())

    def _add_physical(self, cost: dict[str, Any] | None) -> None:
        self._physical_cost = add_physical_cost(self._physical_cost, cost)

    def _append_event(self, value: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
