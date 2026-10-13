"""SSH 命令与文件传输原语。"""

from __future__ import annotations

import shlex
from pathlib import Path, PurePosixPath
from typing import Any


def run(client: Any, command: str, timeout: float = 1900) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), out, err


def ensure_weights(
    client: Any,
    local: Path,
    sha: str,
    remote: PurePosixPath,
) -> None:
    """缓存命中时跳过；否则上传临时文件、校验摘要并原子改名。"""
    code, _, _ = run(client, f"test -f {shlex.quote(str(remote))}", 60)
    if code == 0:
        return
    partial = f"{remote}.partial"
    code, _, err = run(client, f"mkdir -p {shlex.quote(str(remote.parent))}", 60)
    if code:
        raise RuntimeError(err)
    sftp = client.open_sftp()
    try:
        sftp.put(str(local), partial)
    finally:
        sftp.close()
    code, out, err = run(client, f"sha256sum {shlex.quote(partial)}", 600)
    if code or out.split()[:1] != [sha]:
        run(client, f"rm -f {shlex.quote(partial)}", 60)
        raise RuntimeError(f"远端权重校验失败：{err or out}")
    code, _, err = run(client, f"mv -f {shlex.quote(partial)} {shlex.quote(str(remote))}", 60)
    if code:
        raise RuntimeError(err)
