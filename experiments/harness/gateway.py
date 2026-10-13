"""受控 Tool Gateway：唯一的 Agent 文件与 TorchPort 能力边界。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from torchport.contracts import file_hash, load_task
from torchport.diagnosis import inspect_task, localize, trace
from torchport.runtime import (
    add_physical_cost,
    empty_physical_cost,
    execute,
    new_run_dir,
    run_physical_cost,
    write_json,
)
from torchport.verification import build_oracle, evaluate, load_oracle, run_side, verify

from .agents.research import fetch as web_fetch
from .agents.research import search as web_search
from .feedback import filter_feedback, public_run, trace_summary
from .protocol import TrialProtocol
from .tooling import ORACLE_TOOLS, event_arguments, required, tool_schemas, validate_call
from .workspace import WorkspaceAccess

MAX_PROBE_ARGS = 64
MAX_PROBE_ARG_BYTES = 4096
MAX_PROBE_OUTPUT_CHARS = 20_000


class HarnessGateway:
    """一次 trial 的有状态工具接口；最终认证故意不注册成 Agent tool。"""

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
        self._research_cost = {
            "search_calls": 0,
            "fetch_calls": 0,
            "wall_seconds": 0.0,
        }
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
        self.files = WorkspaceAccess(
            self.workspace,
            lambda: Path(self._task()["source"]["repo"]),
            self._candidate_identity,
        )
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
                "external_research_cost": dict(self._research_cost),
                "response_ok": response["ok"],
                "error_type": (response.get("error") or {}).get("type"),
                "response_artifact": str(artifact),
                "response_sha256": file_hash(artifact),
            }
        )
        return response

    def record_agent_event(self, event: str, **facts: Any) -> None:
        """供可信 adapter 记录模型 usage、turn 和停止事件；不暴露为 Agent tool。"""
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
            "external_research_cost": dict(self._research_cost),
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
        report = verify(
            load_task(self.hidden_task_path),
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
            "external_research_cost": dict(self._research_cost),
            "official_wall_seconds": official_wall,
            "tool_calls": self.calls,
            "agent_wall_seconds": agent_wall,
            "candidate_identity": self._candidate_identity(),
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
        if tool == "list_files":
            return self.files.list_files(
                str(arguments.get("path", ".")), str(arguments.get("side", "target"))
            )
        if tool == "read_file":
            return self.files.read_file(
                required(arguments, "path"),
                str(arguments.get("side", "target")),
                arguments.get("start_line"),
                arguments.get("end_line"),
            )
        if tool == "search_code":
            return self.files.search_code(
                required(arguments, "query"),
                str(arguments.get("path", ".")),
                str(arguments.get("side", "target")),
            )
        if tool == "apply_patch":
            return self.files.apply_patch(
                required(arguments, "path"), arguments.get("old"), required(arguments, "new")
            )
        if tool == "inspect":
            return inspect_task(self._task())
        if tool == "probe":
            return self._probe(required(arguments, "path"), arguments.get("args", []))
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
        if tool == "web_search":
            return self._research("search", lambda: web_search(required(arguments, "query")))
        if tool == "web_fetch":
            return self._research("fetch", lambda: web_fetch(required(arguments, "url")))
        raise ValueError(f"未知或未授权工具：{tool}")

    def _research(self, kind: str, action: Callable[[], Any]) -> Any:
        started = self.clock()
        try:
            return action()
        finally:
            self._research_cost[f"{kind}_calls"] += 1
            self._research_cost["wall_seconds"] += max(0.0, self.clock() - started)

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

    def _probe(self, value: str, raw_args: Any) -> dict[str, Any]:
        script = self.files.path(value)
        if script.suffix.lower() != ".py":
            raise ValueError("probe 只允许运行 target 内的 .py 文件")
        if not script.is_file():
            raise FileNotFoundError(value)
        if not isinstance(raw_args, list) or any(not isinstance(item, str) for item in raw_args):
            raise TypeError("probe.args 必须是字符串数组")
        if len(raw_args) > MAX_PROBE_ARGS:
            raise ValueError(f"probe.args 最多允许 {MAX_PROBE_ARGS} 项")
        if any(len(item.encode("utf-8")) > MAX_PROBE_ARG_BYTES for item in raw_args):
            raise ValueError(f"probe.args 单项不能超过 {MAX_PROBE_ARG_BYTES} 字节")

        task = self._task()
        relative = script.relative_to(self.workspace).as_posix()
        run_dir = new_run_dir(self.evidence_root / "probes", self.trial_id, "probe")
        timeout = min(
            float(task["protocol"]["timeout"]),
            float(self.budget_status()["wall_seconds_remaining"]),
        )
        execution = execute(
            task["target"],
            [relative, *raw_args],
            run_dir=run_dir,
            cache=self.evidence_root / "cache" / self.trial_id / "probe-target",
            timeout=timeout,
            module=None,
        )
        cost = run_physical_cost({"execution": execution}, task["target"]["device"])
        self._add_physical(cost)
        stdout, stdout_truncated = _text_tail(run_dir / "stdout.log")
        stderr, stderr_truncated = _text_tail(run_dir / "stderr.log")
        manifest = {
            "schema_version": 1,
            "kind": "torchport.probe",
            "probe_id": run_dir.name,
            "path": relative,
            "args": raw_args,
            "device": task["target"]["device"],
            "candidate_identity": task["target_identity"],
            "execution": execution,
            "physical_cost": cost,
        }
        write_json(run_dir / "probe.json", manifest)
        return {
            "kind": manifest["kind"],
            "probe_id": manifest["probe_id"],
            "path": relative,
            "ok": bool(execution["ok"]),
            "exit_code": execution["exit_code"],
            "timeout": bool(execution["timeout"]),
            "seconds": execution["seconds"],
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "physical_cost": cost,
        }

    def _append_event(self, value: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _text_tail(path: Path) -> tuple[str, bool]:
    text = path.read_text("utf-8", "replace")
    if len(text) <= MAX_PROBE_OUTPUT_CHARS:
        return text, False
    return text[-MAX_PROBE_OUTPUT_CHARS:], True
