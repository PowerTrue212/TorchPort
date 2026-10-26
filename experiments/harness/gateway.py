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
from .protocol import INVALID_TRIAL_STOP_REASONS, FeedbackLevel, TrialProtocol
from .tooling import ORACLE_TOOLS, event_arguments, tool_schemas, validate_call


class HarnessGateway:
    """一次 trial 的有状态 TorchPort 接口；不接管 Agent 的通用代码工具。"""

    def __init__(
        self,
        task: str | Path | Mapping[str, str | Path],
        evidence_root: str | Path,
        trial_dir: str | Path,
        protocol: TrialProtocol,
        *,
        hidden_task: str | Path | Mapping[str, str | Path] | None = None,
        trial_id: str = "t0",
        metadata: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        executor: Any | None = None,
    ) -> None:
        self.task_paths = (
            {name: Path(path).resolve() for name, path in task.items()}
            if isinstance(task, Mapping) else {"default": Path(task).resolve()}
        )
        if not self.task_paths or any(not name or not name.isidentifier() for name in self.task_paths):
            raise ValueError("任务名称必须是非空标识符")
        self.task_path = next(iter(self.task_paths.values()))
        self.hidden_task_paths = (
            {name: Path(path).resolve() for name, path in hidden_task.items()}
            if isinstance(hidden_task, Mapping) else
            {name: Path(hidden_task).resolve() for name in self.task_paths}
            if hidden_task else dict(self.task_paths)
        )
        if set(self.hidden_task_paths) != set(self.task_paths):
            raise ValueError("开发与隐藏任务的名称必须一致")
        self.hidden_task_path = next(iter(self.hidden_task_paths.values()))
        self.executor = executor
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
        self._runs: dict[str, tuple[str, dict[str, Any]]] = {}
        self._last_run_id: str | None = None
        self._last_trace: dict[str, Any] | None = None

        task_value = self._task()
        self.workspace = Path(task_value["target"]["repo"]).resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"candidate 工作区不存在：{self.workspace}")
        for name in self.task_paths:
            for path in (self.task_paths[name], self.hidden_task_paths[name]):
                target = Path(load_task(path)["target"]["repo"]).resolve()
                if target != self.workspace:
                    raise ValueError("所有开发与隐藏 TaskSpec 必须指向同一 candidate 工作区")
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.trial_dir / "events.jsonl"
        self.state_path = self.trial_dir / "gateway-state.json"
        trial = {
            "schema_version": 1,
            "created_at": time.time(),
            "trial_id": trial_id,
            "task": str(self.task_path),
            "hidden_task": str(self.hidden_task_path),
            "task_id": task_value["task_id"],
            "task_digest": task_value["digest"],
            "tasks": {name: str(path) for name, path in self.task_paths.items()},
            "hidden_tasks": {name: str(path) for name, path in self.hidden_task_paths.items()},
            "workspace": str(self.workspace),
            "initial_candidate_identity": task_value["target_identity"],
            "protocol": protocol.as_dict(),
        }
        for key, value in (metadata or {}).items():
            if key in trial:
                raise ValueError(f"trial metadata 不能覆盖保留字段：{key}")
            trial[key] = value
        trial_path = self.trial_dir / "trial.json"
        if trial_path.exists():
            previous = json.loads(trial_path.read_text(encoding="utf-8"))
            for key in ("trial_id", "tasks", "hidden_tasks", "workspace", "protocol"):
                if previous.get(key) != trial.get(key):
                    raise ValueError(f"无法恢复 trial：{key} 已改变")
            if not self.state_path.is_file():
                raise ValueError("已有 trial 缺少状态文件，不能重置预算")
            self.claim_status = previous.get("claim_status", "formal")
            self._load_state()
        else:
            self.claim_status = trial.get("claim_status", "formal")
            write_json(trial_path, trial)
            self._save_state()

    def tool_schemas(self) -> list[dict[str, Any]]:
        return tool_schemas(self.protocol, tuple(self.task_paths) if len(self.task_paths) > 1 else ())

    def call(self, tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """调用一个白名单工具，并追加不可变事件。"""
        arguments = dict(arguments or {})
        started = self.clock()
        before = self._candidate_identity()
        response: dict[str, Any]
        self.calls += 1
        self._save_state()
        try:
            self._check_wall_budget()
            self._validate_call(tool, arguments)
            if tool in ORACLE_TOOLS:
                self._spend_oracle_query(tool)
            response = {"ok": True, "data": self._dispatch(tool, arguments)}
        except Exception as exc:  # 工具边界必须返回结构化错误，不能击穿 Agent loop
            message = str(exc)
            safe_prefixes = (
                "工具在 ", "工具 ", "没有可评估的 run_id", "run_id 与任务不匹配",
                "没有 trace", "没有支持 trace", "wall budget", "oracle query budget",
            )
            if not message.startswith(safe_prefixes):
                write_json(self.trial_dir / "tool-errors" / f"{self.calls:04d}.json", {
                    "type": type(exc).__name__, "message": message,
                })
                message = "验证服务执行失败；详情仅保存在可信控制端"
            response = {
                "ok": False,
                "error": {"type": type(exc).__name__, "message": message},
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
        self._save_state()
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
        if (self.trial_dir / "result.json").exists():
            raise RuntimeError("trial 已经完成认证，不能重复执行")
        official_started = self.clock()
        agent_wall = max(0.0, official_started - self.started)
        delivery = finalize_candidate(
            self.workspace,
            self.trial_dir / "delivery",
            required_entry=self.protocol.candidate_entry,
            resume_existing=True,
        )
        official_reports = {}
        official_candidates = {}
        for name, hidden_path in self.hidden_task_paths.items():
            official_task_value = load_task(hidden_path)
            official_task_value["target"]["repo"] = delivery["snapshot"]
            official_task_path = self.trial_dir / "delivery" / f"official-task-{name}.json"
            write_json(official_task_path, official_task_value)
            official_task = load_task(official_task_path)
            official_candidates[name] = official_task["target_identity"]
            official_reports[name] = self._verify_task(
                name, official_task, official_task_path, official=True
            )
        official_wall = max(0.0, self.clock() - official_started)
        write_json(self.trial_dir / "official-verify.json", official_reports)
        official_physical = empty_physical_cost()
        for report in official_reports.values():
            official_physical = add_physical_cost(official_physical, report.get("physical_cost"))
        total_physical = add_physical_cost(self._physical_cost, official_physical)
        result = {
            "schema_version": 1,
            "trial_id": self.trial_id,
            "passed": all(bool(report.get("passed")) for report in official_reports.values()),
            "valid": (
                stop_reason not in INVALID_TRIAL_STOP_REASONS
                and self.claim_status != "pilot_only"
            ),
            "claim_status": self.claim_status,
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
            "candidate_identity": official_candidates if len(official_candidates) > 1 else next(iter(official_candidates.values())),
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
            return {
                name: self._public_inspect(self._task(name))
                for name in self.task_paths
            } if len(self.task_paths) > 1 else self._public_inspect(self._task())
        if tool == "run":
            name = str(arguments.get("task") or next(iter(self.task_paths)))
            task = self._task(name)
            if self.executor is None:
                oracle = build_oracle(task, self.evidence_root / name)
                if not oracle.get("ok"):
                    self._add_physical(oracle.get("physical_cost"))
                    return self._feedback(oracle)
                manifest = run_side(
                    task, "target", self.evidence_root / name,
                    trial=self.trial_id, oracle=Path(oracle["dir"]),
                )
                cost = add_physical_cost(
                    oracle.get("physical_cost"),
                    run_physical_cost(manifest, task["target"]["device"]),
                )
            else:
                manifest = self.executor.call("run", self.task_paths[name], trial=self.trial_id)
                if "run_id" not in manifest:
                    self._add_physical(manifest.get("physical_cost"))
                    return self._feedback(manifest)
                cost = manifest.get("physical_cost") or run_physical_cost(manifest, task["target"]["device"])
            run_id = f"{name}:{manifest['run_id']}" if len(self.task_paths) > 1 else manifest["run_id"]
            self._runs[run_id] = (name, manifest)
            self._last_run_id = run_id
            self._add_physical(cost)
            return {**public_run(manifest), "run_id": run_id, "task": name, "physical_cost": cost}
        if tool == "evaluate":
            run_id = str(arguments.get("run_id") or self._last_run_id or "")
            if run_id not in self._runs:
                raise ValueError("没有可评估的 run_id；请先调用 run")
            name, manifest = self._runs[run_id]
            if arguments.get("task") and arguments["task"] != name:
                raise ValueError("run_id 与任务不匹配")
            task = self._task(name)
            report = (
                evaluate(load_oracle(task, self.evidence_root / name), manifest,
                         task["protocol"], task["target_identity"])
                if self.executor is None else
                self.executor.call("evaluate", self.task_paths[name], trial=self.trial_id,
                                   manifest=manifest)
            )
            return self._feedback(report)
        if tool == "verify":
            reports = {}
            for name, path in self.task_paths.items():
                report = self._verify_task(name, self._task(name), path)
                self._add_physical(report.get("physical_cost"))
                reports[name] = self._feedback(report)
            if len(reports) == 1:
                return next(iter(reports.values()))
            return {"passed": all(bool(item.get("passed")) for item in reports.values()),
                    "tasks": reports}
        if tool == "trace":
            name = self._trace_task_name()
            self._last_trace = (
                trace(self._task(name), self.evidence_root / name, trial=self.trial_id)
                if self.executor is None else
                self.executor.call("trace", self.task_paths[name], trial=self.trial_id)
            )
            self._add_physical(self._last_trace.get("physical_cost"))
            summary = trace_summary(self._last_trace)
            summary.pop("artifacts", None)
            return {**summary, "task": name}
        if tool == "localize":
            if self._last_trace is None:
                raise ValueError("没有 trace；请先调用 trace")
            return self._strip_private_paths(localize(self._last_trace))
        raise ValueError(f"未知或未授权工具：{tool}")

    def _validate_call(self, tool: str, arguments: dict[str, Any]) -> None:
        validate_call(self.protocol, tool, arguments,
                      tuple(self.task_paths) if len(self.task_paths) > 1 else ())

    def _feedback(self, report: dict[str, Any]) -> dict[str, Any]:
        result = filter_feedback(report, self.protocol.feedback)
        if self.protocol.feedback == FeedbackLevel.LOCALIZED:
            return self._strip_private_paths(result)
        return result

    @staticmethod
    def _strip_private_paths(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: HarnessGateway._strip_private_paths(item)
                for key, item in value.items()
                if key not in {"artifacts", "oracle", "path", "repo", "python",
                               "summary", "next", "reason", "stderr_tail", "traceback"}
            }
        if isinstance(value, list):
            return [HarnessGateway._strip_private_paths(item) for item in value]
        return value

    @staticmethod
    def _public_inspect(task: dict[str, Any]) -> dict[str, Any]:
        result = inspect_task(task)
        result["source"]["repo"] = "<frozen-source>"
        result["target"]["repo"] = "."
        for side in ("source", "target"):
            result[side].pop("python", None)
        result["inputs"] = [
            {key: value for key, value in item.items() if key in {"shape", "dtype"}}
            for item in result["inputs"]
        ]
        return result

    def _task(self, name: str | None = None) -> dict[str, Any]:
        return load_task(self.task_paths[name or next(iter(self.task_paths))])

    def _candidate_identity(self) -> str:
        return "|".join(self._task(name)["target_identity"] for name in self.task_paths)

    def _verify_task(
        self, name: str, task: dict[str, Any], path: Path, *, official: bool = False
    ) -> dict[str, Any]:
        if self.executor is not None:
            return self.executor.call("verify", path, trial=f"{self.trial_id}-official" if official else self.trial_id)
        root = self.evidence_root / ("official" if official else name) / name
        return verify(task, root, trial=f"{self.trial_id}-official" if official else self.trial_id)

    def _trace_task_name(self) -> str:
        for name in self.task_paths:
            if self._task(name)["protocol"]["profile"] != "train-step":
                return name
        raise ValueError("没有支持 trace 的推理任务")

    def _check_wall_budget(self) -> None:
        if self.clock() - self.started >= self.protocol.max_wall_seconds:
            raise TimeoutError("wall budget 已耗尽")

    def _spend_oracle_query(self, tool: str) -> None:
        if self.oracle_queries >= self.protocol.max_oracle_queries:
            raise RuntimeError("oracle query budget 已耗尽")
        self.oracle_queries += 1
        self._queries_by_tool[tool] += 1
        self._queried_candidates.add(self._candidate_identity())
        self._save_state()

    def _add_physical(self, cost: dict[str, Any] | None) -> None:
        self._physical_cost = add_physical_cost(self._physical_cost, cost)

    def _append_event(self, value: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")

    def _save_state(self) -> None:
        write_json(self.state_path, {
            "started_epoch": time.time() - max(0.0, self.clock() - self.started),
            "oracle_queries": self.oracle_queries,
            "queried_candidates": sorted(self._queried_candidates),
            "queries_by_tool": self._queries_by_tool,
            "physical_cost": self._physical_cost,
            "calls": self.calls,
            "runs": {name: {"task": task, "manifest": manifest}
                     for name, (task, manifest) in self._runs.items()},
            "last_run_id": self._last_run_id,
            "last_trace": self._last_trace,
        })

    def _load_state(self) -> None:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.started = self.clock() - max(0.0, time.time() - state["started_epoch"])
        self.oracle_queries = state["oracle_queries"]
        self._queried_candidates = set(state["queried_candidates"])
        self._queries_by_tool = state["queries_by_tool"]
        self._physical_cost = state["physical_cost"]
        self.calls = state["calls"]
        self._runs = {name: (item["task"], item["manifest"])
                      for name, item in state["runs"].items()}
        self._last_run_id = state["last_run_id"]
        self._last_trace = state["last_trace"]
