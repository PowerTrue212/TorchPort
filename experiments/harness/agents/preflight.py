"""Pre-trial isolation probe; run as the Agent account before a session starts.

Standard library only: the Agent account must not be able to import torchport, so
this file cannot depend on it. Every probe must FAIL for the trial to start. The
JSON output is the raw evidence behind the worker attestation, not a sandbox.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _probe_path(path: str) -> dict[str, Any]:
    target = Path(path)
    result: dict[str, Any] = {"path": path}
    try:
        if target.is_dir():
            entries = list(os.scandir(target))
            result.update(readable=True, kind="directory", entries=len(entries))
        else:
            with target.open("rb") as stream:
                stream.read(1)
            result.update(readable=True, kind="file")
    except FileNotFoundError:
        result.update(readable=False, reason="not_found")
    except PermissionError:
        result.update(readable=False, reason="permission_denied")
    except OSError as exc:
        result.update(readable=False, reason=type(exc).__name__)
    return result


def _probe_import(module: str, python: str) -> dict[str, Any]:
    """在独立解释器里试导入，避免本进程的 sys.path 影响结论。"""
    completed = subprocess.run(
        [python, "-c", f"import {module}"], capture_output=True, text=True, timeout=60,
    )
    return {"module": module, "python": python, "importable": completed.returncode == 0}


def _probe_url(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "preflight"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {"url": url, "reachable": True, "status": response.status}
    except urllib.error.HTTPError as exc:
        # 401/403/404 说明私有仓库或过滤生效；其余状态码视为可达
        return {"url": url, "reachable": exc.code not in {401, 403, 404, 451},
                "status": exc.code}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"url": url, "reachable": False, "reason": type(exc).__name__}


def _probe_tcp(address: str, timeout: float) -> dict[str, Any]:
    host, _, port = address.rpartition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return {"address": address, "reachable": True}
    except OSError as exc:
        return {"address": address, "reachable": False, "reason": type(exc).__name__}


def run(config: dict[str, Any]) -> dict[str, Any]:
    python = config.get("python") or sys.executable
    paths = [_probe_path(item) for item in config.get("forbidden_paths", [])]
    imports = [_probe_import(item, python) for item in config.get("forbidden_imports", ["torchport"])]
    timeout = float(config.get("url_timeout", 10))
    urls = [_probe_url(item, timeout) for item in config.get("forbidden_urls", [])]
    # 出口暂不拦截的 URL 只记录可达性，不阻止启动；靠事后审计
    advisory = [_probe_url(item, timeout) for item in config.get("advisory_urls", [])]
    # 公开网络必须可用：协议保留网页与代码检索
    required = [_probe_url(item, timeout) for item in config.get("required_urls", [])]
    # MCP 服务要求 token，HTTP 探测会得到 401，这里只确认 TCP 可连
    tcp = [_probe_tcp(item, timeout) for item in config.get("required_tcp", [])]
    failures = (
        [f"path:{item['path']}" for item in paths if item["readable"]]
        + [f"import:{item['module']}" for item in imports if item["importable"]]
        + [f"url:{item['url']}" for item in urls if item["reachable"]]
        + [f"required_url_unreachable:{item['url']}" for item in required if not item["reachable"]]
        + [f"required_tcp_unreachable:{item['address']}" for item in tcp if not item["reachable"]]
    )
    workspace = config.get("workspace")
    writable = None
    if workspace:
        marker = Path(workspace) / f".preflight-{os.getpid()}"
        try:
            marker.write_text("ok", encoding="utf-8")
            marker.unlink()
            writable = True
        except OSError:
            writable = False
            failures.append(f"workspace_not_writable:{workspace}")
    return {
        "schema_version": 1,
        "kind": "jittor-harness.preflight",
        "created_at": time.time(),
        "user": getpass.getuser(),
        "host": platform.node(),
        "cwd": os.getcwd(),
        "workspace": workspace,
        "workspace_writable": writable,
        "paths": paths,
        "imports": imports,
        "urls": urls,
        "advisory_urls": advisory,
        "required_urls": required,
        "required_tcp": tcp,
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="省略时写到 stdout")
    args = parser.parse_args()
    report = run(json.loads(args.config.read_text(encoding="utf-8")))
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    sys.exit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
