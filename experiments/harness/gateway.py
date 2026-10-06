"""受控 Tool Gateway：唯一的 Agent 文件与 TorchPort 能力边界。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from torchport.diagnosis import inspect_task, localize, trace
from torchport.execution import (
    add_physical_cost,
    empty_physical_cost,
    execute,
    new_run_dir,
    run_physical_cost,
    write_json,
)
from torchport.oracle import build_oracle, evaluate, load_oracle, run_side, verify
from torchport.task import CODE_SUFFIXES, file_hash, load_task

from .protocol import DIAGNOSTIC_FIELDS, FeedbackLevel, TrialProtocol
from .web_research import fetch as web_fetch
from .web_research import search as web_search

ORACLE_TOOLS = {"evaluate", "trace", "verify"}
LOCALIZED_TOOLS = {"trace", "localize"}
CODE_TOOLS = {"list_files", "read_file", "search_code", "apply_patch"}
RESERVED_PARTS = {".git", ".torchport", "__pycache__", "node_modules"}
MAX_TEXT_BYTES = 1_000_000
MAX_READ_LINES = 200
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
        """适配 DSH、Responses API 或其他 function-calling Agent 的稳定工具面。"""
        tools = [
            _tool(
                "list_files",
                "列出只读 source 或可写 target 内文件",
                {"path": "string", "side": {"type": "string", "enum": ["source", "target"]}},
            ),
            _tool(
                "read_file",
                "按行读取 source 或 target 文本文件；每次最多 200 行",
                {
                    "path": "string",
                    "side": {"type": "string", "enum": ["source", "target"]},
                    "start_line": "integer",
                    "end_line": "integer",
                },
                ("path",),
            ),
            _tool(
                "search_code",
                "在 candidate 内搜索纯文本",
                {
                    "query": "string",
                    "path": "string",
                    "side": {"type": "string", "enum": ["source", "target"]},
                },
                ("query",),
            ),
            _tool(
                "apply_patch",
                "精确替换一段文本；old=null 时创建新文件",
                {"path": "string", "old": ["string", "null"], "new": "string"},
                ("path", "old", "new"),
            ),
            _tool("inspect", "查看公开 TaskSpec，不读取 oracle", {}),
            _tool(
                "probe",
                "用冻结的 target Python 运行 candidate 内的临时 .py 自测脚本",
                {
                    "path": "string",
                    "args": {"type": "array", "items": {"type": "string"}},
                },
                ("path",),
            ),
            _tool("run", "运行当前 target candidate", {}),
            _tool("evaluate", "评估最近一次 run", {"run_id": "string"}),
            _tool("verify", "按开发协议重复验证当前 candidate", {}),
            _tool("trace", "采集双端 trace", {}),
            _tool("localize", "定位最近一次 trace 的首次分歧", {}),
        ]
        if self.protocol.allow_network_research:
            tools.extend(
                [
                    _tool("web_search", "搜索公开网络资料", {"query": "string"}, ("query",)),
                    _tool("web_fetch", "读取一个公开网页的正文", {"url": "string"}, ("url",)),
                ]
            )
        if self.protocol.feedback != FeedbackLevel.LOCALIZED:
            tools = [tool for tool in tools if tool["name"] not in LOCALIZED_TOOLS]
        return tools

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
                "arguments": _event_arguments(tool, arguments),
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
            return self._list_files(
                str(arguments.get("path", ".")), str(arguments.get("side", "target"))
            )
        if tool == "read_file":
            return self._read_file(
                _required(arguments, "path"),
                str(arguments.get("side", "target")),
                arguments.get("start_line"),
                arguments.get("end_line"),
            )
        if tool == "search_code":
            return self._search_code(
                _required(arguments, "query"),
                str(arguments.get("path", ".")),
                str(arguments.get("side", "target")),
            )
        if tool == "apply_patch":
            return self._apply_patch(
                _required(arguments, "path"), arguments.get("old"), _required(arguments, "new")
            )
        if tool == "inspect":
            return inspect_task(self._task())
        if tool == "probe":
            return self._probe(_required(arguments, "path"), arguments.get("args", []))
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
            return {**_public_run(manifest), "physical_cost": cost}
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
            return _trace_summary(self._last_trace)
        if tool == "localize":
            if self._last_trace is None:
                raise ValueError("没有 trace；请先调用 trace")
            return localize(self._last_trace)
        if tool == "web_search":
            return self._research("search", lambda: web_search(_required(arguments, "query")))
        if tool == "web_fetch":
            return self._research("fetch", lambda: web_fetch(_required(arguments, "url")))
        raise ValueError(f"未知或未授权工具：{tool}")

    def _research(self, kind: str, action: Callable[[], Any]) -> Any:
        started = self.clock()
        try:
            return action()
        finally:
            self._research_cost[f"{kind}_calls"] += 1
            self._research_cost["wall_seconds"] += max(0.0, self.clock() - started)

    def _validate_call(self, tool: str, arguments: dict[str, Any]) -> None:
        schemas = {item["name"]: item["parameters"] for item in self.tool_schemas()}
        if tool not in schemas:
            raise PermissionError(f"工具在 {self.protocol.feedback.value} 条件中不可用：{tool}")
        schema = schemas[tool]
        properties = schema["properties"]
        extra = sorted(set(arguments) - set(properties))
        if extra:
            raise ValueError(f"工具 {tool} 包含未声明参数：{', '.join(extra)}")
        missing = [name for name in schema["required"] if name not in arguments]
        if missing:
            raise ValueError(f"工具 {tool} 缺少参数：{', '.join(missing)}")
        for name, value in arguments.items():
            allowed = properties[name].get("type")
            allowed = [allowed] if isinstance(allowed, str) else list(allowed or [])
            actual = (
                "null" if value is None else "string" if isinstance(value, str) else
                "integer" if isinstance(value, int) and not isinstance(value, bool) else
                "number" if isinstance(value, (int, float)) and not isinstance(value, bool) else
                "boolean" if isinstance(value, bool) else "object" if isinstance(value, dict) else
                "array" if isinstance(value, list) else "unknown"
            )
            if allowed and actual not in allowed:
                raise TypeError(f"工具 {tool} 参数 {name} 类型应为 {allowed}，实际为 {actual}")
            choices = properties[name].get("enum")
            if choices and value not in choices:
                raise ValueError(f"工具 {tool} 参数 {name} 必须是 {choices}")

    def _feedback(self, report: dict[str, Any]) -> dict[str, Any]:
        if self.protocol.feedback == FeedbackLevel.LOCALIZED:
            return report
        if self.protocol.feedback == FeedbackLevel.VERDICT:
            return {"passed": bool(report.get("passed"))}
        if report.get("kind") == "torchport.verify":
            last = (report.get("results") or [{}])[-1]
            return _diagnostic({**last, **{
                key: report.get(key)
                for key in ("passed", "aligned", "strict_aligned", "mismatch", "failure_code")
                if key in report
            }})
        return _diagnostic(report)

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

    def _root(self, side: str) -> Path:
        if side == "target":
            return self.workspace
        if side == "source":
            return Path(self._task()["source"]["repo"]).resolve()
        raise ValueError("side 必须是 source 或 target")

    def _path(self, value: str, *, side: str = "target", allow_directory: bool = False) -> Path:
        root = self._root(side)
        raw = Path(value)
        if raw.is_absolute():
            raise PermissionError("只允许仓库内的相对路径")
        path = (root / raw).resolve()
        if path != root and root not in path.parents:
            raise PermissionError(f"路径越过 {side} 工作区")
        relative = path.relative_to(root)
        if any(part in RESERVED_PARTS for part in relative.parts):
            raise PermissionError("该路径属于保留目录")
        if not allow_directory and path.exists() and path.is_dir():
            raise IsADirectoryError(value)
        return path

    def _list_files(self, value: str, side: str) -> list[str]:
        side_root = self._root(side)
        root = self._path(value, side=side, allow_directory=True)
        if not root.is_dir():
            raise FileNotFoundError(value)
        files: list[str] = []
        for directory, names, names_files in os.walk(root, followlinks=False):
            names[:] = sorted(name for name in names if name not in RESERVED_PARTS)
            for name in sorted(names_files):
                path = Path(directory) / name
                if path.is_file():
                    files.append(path.relative_to(side_root).as_posix())
                    if len(files) >= 500:
                        return files
        return files

    def _read_file(
        self,
        value: str,
        side: str,
        start_line: int | None,
        end_line: int | None,
    ) -> dict[str, Any]:
        root = self._root(side)
        path = self._path(value, side=side)
        if not path.is_file():
            raise FileNotFoundError(value)
        if path.stat().st_size > MAX_TEXT_BYTES:
            raise ValueError("文件过大")
        lines = path.read_text("utf-8").splitlines(keepends=True)
        total = len(lines)
        if total == 0:
            return {
                "side": side,
                "path": path.relative_to(root).as_posix(),
                "start_line": 1,
                "end_line": 0,
                "total_lines": 0,
                "truncated": False,
                "text": "",
            }
        start = 1 if start_line is None else start_line
        end = min(total, start + MAX_READ_LINES - 1) if end_line is None else end_line
        if start < 1 or end < start:
            raise ValueError("read_file 行号必须满足 1 <= start_line <= end_line")
        if end - start + 1 > MAX_READ_LINES:
            raise ValueError(f"read_file 每次最多读取 {MAX_READ_LINES} 行")
        return {
            "side": side,
            "path": path.relative_to(root).as_posix(),
            "start_line": start,
            "end_line": min(end, total),
            "total_lines": total,
            "truncated": end < total,
            "text": "".join(lines[start - 1 : end]),
        }

    def _search_code(self, query: str, value: str, side: str) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query 不能为空")
        side_root = self._root(side)
        root = self._path(value, side=side, allow_directory=True)
        paths = [root] if root.is_file() else [Path(directory) / name for directory, names, files in os.walk(root, followlinks=False) for name in files if not any(part in RESERVED_PARTS for part in Path(directory).relative_to(side_root).parts)]
        matches: list[dict[str, Any]] = []
        for path in paths:
            if path.suffix.lower() not in CODE_SUFFIXES or path.stat().st_size > MAX_TEXT_BYTES:
                continue
            try:
                lines = path.read_text("utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, 1):
                if query in line:
                    matches.append({"side": side, "path": path.relative_to(side_root).as_posix(), "line": line_number, "text": line[:500]})
                    if len(matches) >= 100:
                        return matches
        return matches

    def _apply_patch(self, value: str, old: Any, new: str) -> dict[str, Any]:
        if not isinstance(new, str):
            raise TypeError("new 必须是字符串")
        path = self._path(value)
        if old is None:
            if path.exists():
                raise FileExistsError(value)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new, "utf-8")
            action = "created"
        else:
            if not isinstance(old, str) or not old:
                raise ValueError("old 必须是非空字符串，创建文件时使用 null")
            if not path.is_file():
                raise FileNotFoundError(value)
            text = path.read_text("utf-8")
            count = text.count(old)
            if count != 1:
                raise ValueError(f"old 必须精确出现一次，实际为 {count} 次")
            path.write_text(text.replace(old, new, 1), "utf-8")
            action = "updated"
        return {
            "path": path.relative_to(self.workspace).as_posix(),
            "action": action,
            "candidate_identity": self._candidate_identity(),
        }

    def _probe(self, value: str, raw_args: Any) -> dict[str, Any]:
        script = self._path(value)
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


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                key: {"type": value} if isinstance(value, (str, list)) else value
                for key, value in properties.items()
            },
            "required": list(required),
            "additionalProperties": False,
        },
    }


def _required(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ValueError(f"缺少字符串参数：{name}")
    return value


def _event_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool != "apply_patch":
        return arguments
    return {
        "path": arguments.get("path"),
        "old_bytes": len((arguments.get("old") or "").encode("utf-8")),
        "new_bytes": len((arguments.get("new") or "").encode("utf-8")),
    }


def _text_tail(path: Path) -> tuple[str, bool]:
    text = path.read_text("utf-8", "replace")
    if len(text) <= MAX_PROBE_OUTPUT_CHARS:
        return text, False
    return text[-MAX_PROBE_OUTPUT_CHARS:], True


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    execution = run.get("execution") or {}
    return {
        "kind": run.get("kind"),
        "run_id": run.get("run_id"),
        "ok": bool(run.get("ok")),
        "mode": run.get("mode"),
        "execution": {
            key: execution.get(key)
            for key in ("ok", "exit_code", "timeout", "seconds", "stderr_tail")
            if key in execution
        },
        "capture": run.get("capture"),
    }


def _trace_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Keep model-visible trace evidence small; the full report stays in artifacts."""
    def comparison(name: str) -> dict[str, Any] | None:
        value = report.get(name)
        if not isinstance(value, dict):
            return None
        return {
            key: value.get(key)
            for key in (
                "passed",
                "profile",
                "first_divergence",
                "first_gating_divergence",
                "metrics",
            )
            if key in value
        }

    return {
        key: report.get(key)
        for key in (
            "kind",
            "task_id",
            "task_digest",
            "trial_id",
            "passed",
            "status",
            "failure_code",
            "stage",
        )
        if key in report
    } | {
        "output": comparison("output"),
        "strict": comparison("strict"),
        "physical_cost": report.get("physical_cost"),
        "artifacts": report.get("artifacts"),
    }


def _diagnostic(report: dict[str, Any]) -> dict[str, Any]:
    failure = str(report.get("failure_code") or "")
    codes = {
        "PARITY_FAILED": "NUMERICAL_MISMATCH",
        "STRUCTURE_MISMATCH": "STRUCTURAL_MISMATCH",
        "RUN_FAILED": "RUN_FAILED",
        "INPUT_MISMATCH": "INPUT_MISMATCH",
        "WEIGHT_LOAD_FAILED": "WEIGHT_LOAD_FAILED",
        "NOT_COMPARED": "NOT_EVALUATED",
        "TOO_FEW_RUNS": "INCOMPLETE_VERIFICATION",
    }
    code = codes.get(failure, "VERIFICATION_FAILED" if failure else None)
    structural = report.get("structural", "not_run")
    evidence = report.get("evidence") or {}
    weights = evidence.get("weights") or {}
    weight_load = "pass" if weights.get("complete") else (
        "fail" if weights else "not_run"
    )
    if report.get("runnable") == "fail" or code == "RUN_FAILED":
        stage = "runnable"
    elif structural == "fail" or code == "STRUCTURAL_MISMATCH":
        stage = "structural"
    elif report.get("aligned") is False or code == "NUMERICAL_MISMATCH":
        stage = "numerical"
    elif report.get("passed"):
        stage = "passed"
    else:
        stage = "verification"
    mismatch = report.get("mismatch") or {}
    metrics = mismatch.get("fp32") or ((report.get("profiles") or {}).get("fp32") or {}).get(
        "metrics", {}
    )
    category = {
        "runnable": "runtime",
        "structural": "structural",
        "numerical": "numeric",
        "verification": "protocol",
        "passed": None,
    }[stage]
    result = {
        "passed": bool(report.get("passed")),
        "aligned": report.get("aligned"),
        "strict_aligned": report.get("strict_aligned"),
        "stage": stage,
        "failure_code": code,
        "category": category,
        "max_abs": metrics.get("max_absolute_error"),
        "mismatch_ratio": metrics.get("mismatch_ratio"),
        "mismatched_elements": metrics.get("mismatched_elements"),
        "mismatched_tensors": metrics.get("mismatched_tensors"),
        "structural": structural,
        "weight_load": weight_load,
    }
    if tuple(result) != DIAGNOSTIC_FIELDS:
        raise AssertionError("diagnostic feedback allowlist drifted")
    return result
