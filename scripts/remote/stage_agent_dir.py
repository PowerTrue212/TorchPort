#!/usr/bin/env python3
"""把冻结输入和官方权重以硬链接放入 Agent 的远端工作目录。"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from dsh_torchport import (
    REMOTE_SOURCE_PYTHON,
    _cached_weights_path,
    _inside_workspace,
    _local_weights,
    _remote_base,
)
from ssh_exec import connect

AGENT_ROOT = PurePosixPath("/root/autodl-tmp/agent")
WORKSPACE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _run(client: Any, command: str, timeout: float = 600) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), out, err


def _write_remote(client: Any, path: PurePosixPath, content: str) -> None:
    stdin, stdout, stderr = client.exec_command(f"cat > {shlex.quote(str(path))}", timeout=60)
    stdin.write(content.encode("utf-8"))
    stdin.flush()
    stdin.channel.shutdown_write()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"写入 {path.name} 失败：{err or out}")


def _remote_shape(client: Any, path: PurePosixPath) -> tuple[list[int], str]:
    script = (
        "import json,numpy as np,sys;"
        "value=np.load(sys.argv[1],mmap_mode='r',allow_pickle=False);"
        "print(json.dumps({'shape':list(value.shape),'dtype':str(value.dtype)}))"
    )
    command = " ".join(
        [
            shlex.quote(REMOTE_SOURCE_PYTHON),
            "-c",
            shlex.quote(script),
            shlex.quote(str(path)),
        ]
    )
    code, out, err = _run(client, command, 120)
    if code:
        raise RuntimeError(f"读取 {path.name} 的 numpy 头失败：{err or out}")
    metadata = json.loads(out.strip())
    return metadata["shape"], metadata["dtype"]


def _sources(task_path: Path, task: dict[str, Any]) -> list[tuple[str, PurePosixPath, bool]]:
    base = _remote_base(task_path)
    sources = []
    for index, item in enumerate(task.get("inputs") or []):
        local = _inside_workspace(item["path"])
        sources.append((local.name, base / "inputs" / f"{index}-{local.name}", True))
    if weights := _local_weights(task_path, task):
        local, sha = weights
        sources.append((local.name, _cached_weights_path(local, sha), False))
    return sources


def stage(workspace_name: str, task_argument: str | Path) -> None:
    if not WORKSPACE_NAME.fullmatch(workspace_name) or workspace_name in {".", ".."}:
        raise ValueError("workspace-name 只能包含字母、数字、点、下划线和连字符")

    task_path = _inside_workspace(task_argument)
    task: dict[str, Any] = json.loads(task_path.read_text("utf-8"))
    sources = _sources(task_path, task)
    names = [name.casefold() for name, _, _ in sources]
    if len(names) != len(set(names)):
        raise ValueError("冻结输入与权重存在同名文件，无法安全放入 staged/")

    staged = AGENT_ROOT / workspace_name / "staged"
    client = connect()
    try:
        for name, source, _ in sources:
            code, _, _ = _run(client, f"test -f {shlex.quote(str(source))}", 60)
            if code:
                raise FileNotFoundError(
                    f"远端尚未准备 {name}；请先用同一个 task 路径运行 "
                    "torchport_inspect 或完成一次 verify。"
                )

        code, out, err = _run(client, f"mkdir -p {shlex.quote(str(staged))}", 60)
        if code:
            raise RuntimeError(f"创建 staged 目录失败：{err or out}")

        lines = [
            "TorchPort 为本次迁移预置的冻结材料",
            "这些文件是远端缓存的硬链接；可用于 Agent 自己的 PyTorch/Jittor 对比。",
            "",
        ]
        for name, source, is_numpy in sources:
            target = staged / name
            command = f"ln -f {shlex.quote(str(source))} {shlex.quote(str(target))}"
            code, out, err = _run(client, command, 60)
            if code:
                raise RuntimeError(f"创建 {name} 的硬链接失败：{err or out}")

            code, out, err = _run(client, f"sha256sum {shlex.quote(str(target))}", 600)
            if code or not out.split():
                raise RuntimeError(f"计算 {name} 的 sha256 失败：{err or out}")
            sha256 = out.split()[0]
            if is_numpy:
                shape, dtype = _remote_shape(client, target)
                shape_text = "x".join(str(value) for value in shape)
                lines.append(f"{name}\tshape={shape_text}\tdtype={dtype}\tsha256={sha256}")
            else:
                lines.append(f"{name}\tshape=-\tsha256={sha256}")

        _write_remote(client, staged / "README.txt", "\n".join(lines) + "\n")
    finally:
        client.close()

    print(f"staged {len(sources)} file(s) in {staged}")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        print("用法：stage_agent_dir.py <workspace-name> <task.json>")
        return 2
    try:
        stage(arguments[0], arguments[1])
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
        print(f"准备失败：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
