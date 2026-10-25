"""Dependency-free MCP adapter; run only in the trusted controller process.

The Agent retains its native code tools. This server exposes only the Gateway's
feedback tools and never accepts arbitrary filesystem paths or shell commands.
Formal trials use the loopback streamable-http transport started by the controller
account; stdio is kept for local self-tests only.
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def _handle(request: Any, gateway: HarnessGateway) -> dict[str, Any] | None:
    try:
        if not isinstance(request, dict):
            return None
        return _response(request, gateway)
    except Exception:
        return {"jsonrpc": "2.0", "id": request.get("id") if isinstance(request, dict) else None,
                "error": {"code": -32603, "message": "Internal server error"}}


def serve(gateway: HarnessGateway) -> None:
    """stdio 传输：只用于本机自测，Agent 会以自身权限拉起该进程。"""
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _handle(request, gateway)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def make_http_server(
    gateway: HarnessGateway, host: str, port: int, token: str
) -> ThreadingHTTPServer:
    """streamable-http 传输：由控制端账户启动，Agent 只经回环地址和 token 访问。"""
    if ipaddress.ip_address(host).is_loopback is False:
        raise ValueError("MCP HTTP 服务只允许监听回环地址")
    if len(token) < 32:
        raise ValueError("MCP token 至少 32 个字符")
    lock = threading.Lock()  # gateway 的预算与事件序号不是线程安全的

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: Any) -> None:  # 不把请求写到控制台
            pass

        def _send(self, status: int, body: bytes = b"", content_type: str | None = None) -> None:
            self.send_response(status)
            if content_type:
                self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _authorized(self) -> bool:
            if self.path.split("?", 1)[0] != "/mcp":
                self._send(404)
                return False
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
                self._send(401)
                return False
            return True

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"null")
            except (ValueError, json.JSONDecodeError):
                self._send(400)
                return
            batch = payload if isinstance(payload, list) else [payload]
            with lock:
                responses = [r for r in (_handle(item, gateway) for item in batch) if r is not None]
            if not responses:
                self._send(202)
                return
            value = responses if isinstance(payload, list) else responses[0]
            self._send(200, json.dumps(value, ensure_ascii=False).encode("utf-8"),
                       "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if self._authorized():
                self._send(405)  # 不提供服务端推送流

        def do_DELETE(self) -> None:  # noqa: N802
            if self._authorized():
                self._send(405)

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--http", metavar="HOST:PORT",
                        help="以 streamable-http 监听回环地址；省略时用 stdio")
    parser.add_argument("--token-file", type=Path,
                        help="控制端保存的 bearer token 文件；--http 时必填")
    args = parser.parse_args()
    gateway = make_gateway(args.trial)
    if args.http is None:
        serve(gateway)
        return
    if args.token_file is None:
        parser.error("--http 需要 --token-file")
    host, _, port = args.http.rpartition(":")
    token = args.token_file.read_text(encoding="utf-8").strip()
    server = make_http_server(gateway, host, int(port), token)
    print(json.dumps({"listening": f"http://{host}:{server.server_address[1]}/mcp"}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
