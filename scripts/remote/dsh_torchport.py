#!/usr/bin/env python3
"""在本地 DSH 中调用远端 GPU 上的同名 TorchPort 命令。"""

from __future__ import annotations

import hashlib
import io
import json
import shlex
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from ssh_exec import connect

REMOTE_ROOT = PurePosixPath("/root/autodl-tmp/tp-dsh-web")
REMOTE_SOURCE_PYTHON = "/root/miniconda3/bin/python"
REMOTE_TARGET_PYTHON = "/root/autodl-tmp/jt-env/bin/python"
REMOTE_PYTHON_CONFIG = "/root/miniconda3/bin/python3.12-config"
SYSTEM_LIBSTDCXX = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
SKIP_PARTS = {".git", ".torchport", "__pycache__", ".pytest_cache", "node_modules"}
TASK_COMMANDS = {"inspect", "run", "evaluate", "trace", "verify"}
COMMANDS = TASK_COMMANDS | {"localize"}


def _inside_workspace(path: str | Path) -> Path:
    root = Path.cwd().resolve()
    value = Path(path)
    value = value.resolve() if value.is_absolute() else (root / value).resolve()
    if value != root and root not in value.parents:
        raise PermissionError(f"路径不在 TorchPort 工作区内：{value}")
    return value


def _remote_base(task_path: Path) -> PurePosixPath:
    digest = hashlib.sha256(str(task_path).encode()).hexdigest()[:12]
    return REMOTE_ROOT / digest


def _report_base(report: str) -> PurePosixPath:
    path = PurePosixPath(report)
    try:
        relative = path.relative_to(REMOTE_ROOT)
    except ValueError as exc:
        raise PermissionError("localize 只接受本次远端实验生成的报告") from exc
    if len(relative.parts) < 2:
        raise PermissionError("localize 报告路径缺少任务目录")
    return REMOTE_ROOT / relative.parts[0]


def _add_tree(package: tarfile.TarFile, path: Path, arcname: str) -> None:
    def allowed(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        return None if any(part in SKIP_PARTS or part.endswith(".pyc") for part in parts) else info

    package.add(path, arcname=arcname, recursive=True, filter=allowed)


def _build_bundle(task_path: Path, base: PurePosixPath) -> bytes:
    task: dict[str, Any] = json.loads(task_path.read_text("utf-8"))
    source = _inside_workspace(task["source"]["repo"])
    target = _inside_workspace(task["target"]["repo"])
    package_root = Path.cwd().resolve() / "src" / "torchport"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)

    rewritten = json.loads(json.dumps(task))
    rewritten["source"]["repo"] = str(base / "source")
    rewritten["source"]["python"] = REMOTE_SOURCE_PYTHON
    rewritten["target"]["repo"] = str(base / "target")
    rewritten["target"]["python"] = REMOTE_TARGET_PYTHON
    target_env = dict(rewritten["target"].get("env") or {})
    target_env.update(
        {
            "LD_PRELOAD": SYSTEM_LIBSTDCXX,
            "PATH": "/usr/local/cuda/bin:/root/autodl-tmp/jt-env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "python_config_path": REMOTE_PYTHON_CONFIG,
            "use_cutlass": "0",
        }
    )
    rewritten["target"]["env"] = target_env

    inputs: list[tuple[Path, str]] = []
    for index, item in enumerate(rewritten.get("inputs") or []):
        local = _inside_workspace(item["path"])
        remote = f"inputs/{index}-{local.name}"
        item["path"] = str(base / remote)
        inputs.append((local, remote))

    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as package:
        _add_tree(package, source, "source")
        _add_tree(package, target, "target")
        _add_tree(package, package_root, "src/torchport")
        for local, remote in inputs:
            package.add(local, arcname=remote)
        payload = json.dumps(rewritten, ensure_ascii=False, indent=2).encode()
        info = tarfile.TarInfo("task.json")
        info.size = len(payload)
        package.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def _run(client: Any, command: str, timeout: float = 1900) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), out, err


def _sync(client: Any, task_path: Path, base: PurePosixPath) -> None:
    archive = _build_bundle(task_path, base)
    remote_archive = str(base) + ".tar.gz"
    code, _, err = _run(client, f"mkdir -p {shlex.quote(str(REMOTE_ROOT))}", 60)
    if code:
        raise RuntimeError(err)
    sftp = client.open_sftp()
    try:
        with sftp.file(remote_archive, "wb") as output:
            output.write(archive)
    finally:
        sftp.close()
    replace = " ".join(
        shlex.quote(str(base / name))
        for name in ("source", "target", "src", "inputs", "task.json")
    )
    command = (
        f"mkdir -p {shlex.quote(str(base))} && rm -rf {replace} && "
        f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(str(base))} && "
        f"rm -f {shlex.quote(remote_archive)}"
    )
    code, _, err = _run(client, command, 120)
    if code:
        raise RuntimeError(err)


def _remote_arguments(arguments: list[str], base: PurePosixPath) -> list[str]:
    command = arguments[0]
    values = list(arguments)
    if command in TASK_COMMANDS:
        values[1] = str(base / "task.json")
        if "--root" in values:
            index = values.index("--root")
            values[index + 1] = str(base / "evidence")
    return values


def main(arguments: list[str] | None = None) -> int:
    arguments = list(arguments or sys.argv[1:])
    if not arguments or arguments[0] not in COMMANDS:
        raise SystemExit("仅支持 inspect/run/evaluate/trace/localize/verify")
    task_path = _inside_workspace(arguments[1]) if arguments[0] in TASK_COMMANDS else None
    base = _remote_base(task_path) if task_path else _report_base(arguments[1])
    client = connect()
    try:
        if task_path:
            _sync(client, task_path, base)
        remote_args = _remote_arguments(arguments, base)
        quoted = " ".join(shlex.quote(value) for value in remote_args)
        command = (
            f"cd {shlex.quote(str(base))} && "
            f"export PYTHONPATH={shlex.quote(str(base / 'src'))} && "
            f"export LD_PRELOAD={shlex.quote(SYSTEM_LIBSTDCXX)} && "
            f"{REMOTE_SOURCE_PYTHON} -m torchport {quoted}"
        )
        code, out, err = _run(client, command)
        sys.stdout.write(out)
        sys.stderr.write(err)
        return code
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
