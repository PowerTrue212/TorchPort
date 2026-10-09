#!/usr/bin/env python3
"""Agent 侧远端入口。

这只是阻止直接访问敏感路径的子字符串检查，不是安全沙箱。远端 root 用户仍可用
间接路径、解释器或其他方式绕过，因此实验结束后仍必须审计本地调用日志。
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORBIDDEN_SUBSTRINGS = (
    "tp-work",
    "evidence",
    "weights-cache",
    "oracle",
    "envelope",
    ".tp_ssh",
    "/root/autodl-tmp/heldout",
    "/root/autodl-tmp/tp-",
)
AGENT_ROOT = "/root/autodl-tmp/agent"
AGENT_DIRECTORY = re.compile(
    r"/root/autodl-tmp/agent/([^/\s'\";&|()<>]+)", re.IGNORECASE
)
SECRET_VALUE = re.compile(
    r"(?i)\b(password|token|secret|api[_-]?key)\b(\s*=\s*|\s+)([^\s;&|]+)"
)


def _reject(reason: str) -> int:
    print(f"拒绝执行：{reason}")
    return 2


def _audit_path() -> Path:
    configured = os.environ.get("TP_AGENT_AUDIT")
    return Path(configured) if configured else Path.cwd() / ".torchport" / "agent_ssh.jsonl"


def _redact(value: str) -> str:
    """只保留审计所需参数，遮盖常见的内联凭据写法。"""
    return SECRET_VALUE.sub(lambda match: f"{match.group(1)}=<redacted>", value)


def _append_audit(record: dict[str, Any]) -> None:
    path = _audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _allowed_directory() -> str | None:
    raw = os.environ.get("TP_AGENT_REMOTE_DIR", "").strip()
    if not raw:
        return None
    return posixpath.normpath(raw)


def _connect() -> Any:
    # 先完成本地拒绝检查，再导入会读取 SSH 配置的模块。
    from ssh_exec import connect

    return connect()


def _forbidden_command(command: str, allowed: str) -> str | None:
    folded = command.casefold()
    for fragment in FORBIDDEN_SUBSTRINGS:
        if fragment.casefold() in folded:
            return f"命令包含禁止访问的路径标记 {fragment}"

    allowed_prefix = f"{AGENT_ROOT}/"
    allowed_workspace = None
    if allowed.casefold().startswith(allowed_prefix.casefold()):
        allowed_workspace = allowed[len(allowed_prefix) :].split("/", 1)[0].casefold()
    for match in AGENT_DIRECTORY.finditer(command):
        if allowed_workspace is None or match.group(1).casefold() != allowed_workspace:
            return "命令引用了其他 Agent 的远端目录"
    return None


def _resolve_remote_path(remote: str, allowed: str) -> str | None:
    resolved = posixpath.normpath(
        remote if posixpath.isabs(remote) else posixpath.join(allowed, remote)
    )
    prefix = allowed.rstrip("/") + "/"
    if resolved.startswith(prefix):
        return resolved
    return None


def _run(command: str, allowed: str) -> int:
    if reason := _forbidden_command(command, allowed):
        return _reject(reason)

    remote_command = f"cd {shlex.quote(allowed)} && {command}"
    client = _connect()
    try:
        _, stdout, stderr = client.exec_command(remote_command)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
        sys.stdout.write(out)
        sys.stderr.write(err)
        sys.stdout.write(f"\n[remote exit code: {code}]\n")
        sys.stdout.flush()
        return code
    finally:
        client.close()


def _put(local_text: str, remote_text: str, allowed: str) -> int:
    remote = _resolve_remote_path(remote_text, allowed)
    if remote is None:
        return _reject("上传目标不在允许的远端工作目录内")

    local = Path(local_text)
    if not local.is_file():
        return _reject(f"本地文件不存在：{local}")

    command = (
        f"mkdir -p {shlex.quote(posixpath.dirname(remote))} && "
        f"cat > {shlex.quote(remote)}"
    )
    client = _connect()
    try:
        stdin, stdout, stderr = client.exec_command(command)
        with local.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                stdin.write(block)
        stdin.flush()
        stdin.channel.shutdown_write()
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
        sys.stdout.write(out)
        sys.stderr.write(err)
        if code == 0:
            print(f"uploaded {local} -> {remote}")
        return code
    finally:
        client.close()


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    subcommand = arguments[0] if arguments else None
    record: dict[str, Any] = {"subcommand": subcommand, "accepted": False}
    if subcommand == "run" and len(arguments) >= 2:
        record["command"] = _redact(arguments[1])
    elif subcommand == "put" and len(arguments) >= 3:
        record["local"] = str(Path(arguments[1]))
        record["remote"] = arguments[2]
    exit_code = 1
    try:
        allowed = _allowed_directory()
        if allowed is None:
            exit_code = _reject("未设置 TP_AGENT_REMOTE_DIR")
            return exit_code

        record["allowed_dir"] = allowed
        if subcommand == "run" and len(arguments) == 2:
            if reason := _forbidden_command(arguments[1], allowed):
                exit_code = _reject(reason)
                return exit_code
            record["accepted"] = True
            exit_code = _run(arguments[1], allowed)
            return exit_code
        if subcommand == "put" and len(arguments) == 3:
            if _resolve_remote_path(arguments[2], allowed) is None:
                exit_code = _reject("上传目标不在允许的远端工作目录内")
                return exit_code
            if not Path(arguments[1]).is_file():
                exit_code = _reject(f"本地文件不存在：{arguments[1]}")
                return exit_code
            record["accepted"] = True
            exit_code = _put(arguments[1], arguments[2], allowed)
            return exit_code

        exit_code = _reject("只支持 run <命令> 和 put <本地文件> <远端路径>")
        return exit_code
    finally:
        record["exit_code"] = exit_code
        _append_audit(record)


if __name__ == "__main__":
    raise SystemExit(main())
