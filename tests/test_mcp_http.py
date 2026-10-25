from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from typing import Any

from experiments.harness.agents.mcp_server import make_http_server

TOKEN = "t" * 40


class _Gateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [{"name": "inspect", "description": "d", "parameters": {"type": "object"}}]

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool, arguments))
        return {"ok": True, "data": {"tool": tool}}


class McpHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gateway = _Gateway()
        self.server = make_http_server(self.gateway, "127.0.0.1", 0, TOKEN)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/mcp"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _post(self, body: Any, token: str | None = TOKEN, path: str = "/mcp") -> tuple[int, Any]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            self.url.replace("/mcp", path), data=json.dumps(body).encode(), headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                raw = response.read()
                return response.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            return exc.code, None

    def test_rejects_missing_or_wrong_token(self) -> None:
        request = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
        self.assertEqual(self._post(request, token=None)[0], 401)
        self.assertEqual(self._post(request, token="x" * 40)[0], 401)
        self.assertEqual(self.gateway.calls, [])

    def test_unknown_path_is_404(self) -> None:
        self.assertEqual(self._post({"jsonrpc": "2.0", "id": 1, "method": "ping"},
                                    path="/other")[0], 404)

    def test_tools_list_and_call(self) -> None:
        status, listed = self._post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual(status, 200)
        self.assertEqual(listed["result"]["tools"][0]["name"], "inspect")
        status, called = self._post({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "inspect", "arguments": {}},
        })
        self.assertEqual(status, 200)
        self.assertFalse(called["result"]["isError"])
        self.assertEqual(self.gateway.calls, [("inspect", {})])

    def test_notification_returns_202(self) -> None:
        status, _ = self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.assertEqual(status, 202)

    def test_refuses_non_loopback_and_short_token(self) -> None:
        with self.assertRaises(ValueError):
            make_http_server(self.gateway, "0.0.0.0", 0, TOKEN)
        with self.assertRaises(ValueError):
            make_http_server(self.gateway, "127.0.0.1", 0, "short")


if __name__ == "__main__":
    unittest.main()
