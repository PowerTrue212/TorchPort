"""Dependency-free stdio MCP adapter; run only in the trusted controller process.

The Agent retains its native code tools. This server exposes only the Gateway's
feedback tools and never accepts arbitrary filesystem paths or shell commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _project = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_project), str(_project / "src")]

from experiments.harness.gateway import HarnessGateway  # noqa: E402
from experiments.harness.protocol import FeedbackLevel, TrialProtocol  # noqa: E402
from experiments.harness.remote_executor import RemoteExecutor  # noqa: E402


def make_gateway(config_path: Path) -> HarnessGateway:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "pilot_only":
        raise ValueError("MCP 适配层仅支持 pilot_only；正式试验必须经隔离 worker 编排")
    project = Path(config["project_root"]).resolve()
    workspace = Path(config["candidate_root"]).resolve()
    trial_dir = Path(config["trial_dir"]).resolve()
    tasks = {name: Path(path) for name, path in config["tasks"].items()}
    hidden = {name: Path(path) for name, path in config.get("hidden_tasks", config["tasks"]).items()}
    protocol = TrialProtocol(
        feedback=FeedbackLevel(config["feedback"]),
        max_oracle_queries=int(config["max_oracle_queries"]),
        max_wall_seconds=float(config["max_wall_seconds"]),
        candidate_entry=config.get("candidate_entry"),
    )
    executor = RemoteExecutor(project, trial_dir / "remote-raw", workspace, config.get("remote"))
    gateway = HarnessGateway(
        tasks, Path(config["evidence_root"]), trial_dir, protocol,
        hidden_task=hidden, trial_id=config["trial_id"], executor=executor,
        metadata={"agent": config.get("agent", "DSH"), "condition": config["feedback"],
                  "claim_status": "pilot_only"},
    )
    if gateway.workspace != workspace:
        raise ValueError("候选目录与 TaskSpec 不一致")
    return gateway


def _response(request: dict[str, Any], gateway: HarnessGateway) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    if request_id is None:
        return None
    if method == "initialize":
        value = {
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "migration-check", "version": "0.1.0"},
        }
    elif method == "ping":
        value = {}
    elif method == "tools/list":
        value = {"tools": [
            {"name": item["name"], "description": item["description"],
             "inputSchema": item["parameters"]}
            for item in gateway.tool_schemas()
        ]}
    elif method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        result = gateway.call(name, arguments)
        value = {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
            "isError": not result["ok"],
        }
    else:
        return {"jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32601, "message": "Method not found"}}
    return {"jsonrpc": "2.0", "id": request_id, "result": value}


def serve(gateway: HarnessGateway) -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                continue
            response = _response(request, gateway)
        except Exception:
            response = {"jsonrpc": "2.0", "id": request.get("id") if isinstance(request, dict) else None,
                        "error": {"code": -32603, "message": "Internal server error"}}
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, required=True)
    args = parser.parse_args()
    serve(make_gateway(args.trial))


if __name__ == "__main__":
    main()
