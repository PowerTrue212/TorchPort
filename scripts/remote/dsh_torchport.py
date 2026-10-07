#!/usr/bin/env python3
"""在本地 DSH 中调用远端 GPU 上的同名 TorchPort 命令。"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from ssh_exec import connect

REMOTE_ENV_NAMES = {
    "TP_REMOTE_ROOT",
    "TP_REMOTE_SOURCE_PYTHON",
    "TP_REMOTE_TARGET_PYTHON",
    "TP_REMOTE_PYTHON_CONFIG",
    "TP_REMOTE_LIBSTDCXX",
    "TP_REMOTE_TARGET_PATH",
    "TP_REMOTE_NVCC_PATH",
    "TP_REMOTE_USE_CUTT",
    "TP_REMOTE_USE_MKL",
}


def load_env_file(path: str | Path | None = None) -> None:
    """Load remote path settings without exposing unrelated values or secrets."""
    path = Path(path or Path.home() / ".tp_ssh.env")
    if not path.is_file():
        return
    for raw in path.read_text("utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name in REMOTE_ENV_NAMES:
            os.environ.setdefault(name, value)


load_env_file()
REMOTE_ROOT = PurePosixPath(os.environ.get("TP_REMOTE_ROOT", "/root/autodl-tmp/tp-dsh-web"))
REMOTE_SOURCE_PYTHON = os.environ.get("TP_REMOTE_SOURCE_PYTHON", "/root/miniconda3/bin/python")
REMOTE_TARGET_PYTHON = os.environ.get(
    "TP_REMOTE_TARGET_PYTHON", "/root/autodl-tmp/jt-env/bin/python"
)
REMOTE_PYTHON_CONFIG = os.environ.get(
    "TP_REMOTE_PYTHON_CONFIG", "/root/miniconda3/bin/python3.12-config"
)
SYSTEM_LIBSTDCXX = os.environ.get(
    "TP_REMOTE_LIBSTDCXX", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
)
REMOTE_TARGET_PATH = os.environ.get(
    "TP_REMOTE_TARGET_PATH",
    "/usr/local/cuda/bin:/root/autodl-tmp/jt-env/bin:/usr/local/sbin:"
    "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
)
REMOTE_TARGET_ENV_DEFAULTS = {
    "use_cutt": os.environ.get("TP_REMOTE_USE_CUTT"),
    "use_mkl": os.environ.get("TP_REMOTE_USE_MKL"),
    "nvcc_path": os.environ.get("TP_REMOTE_NVCC_PATH"),
}
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
    target_env.update({"PATH": REMOTE_TARGET_PATH, "use_cutlass": "0"})
    for name, value in (
        ("LD_PRELOAD", SYSTEM_LIBSTDCXX),
        ("python_config_path", REMOTE_PYTHON_CONFIG),
    ):
        if value:
            target_env[name] = value
        else:
            target_env.pop(name, None)
    for name, value in REMOTE_TARGET_ENV_DEFAULTS.items():
        if value:
            target_env.setdefault(name, value)
    rewritten["target"]["env"] = target_env

    inputs: list[tuple[Path, str]] = []
    for index, item in enumerate(rewritten.get("inputs") or []):
        local = _inside_workspace(item["path"])
        remote = f"inputs/{index}-{local.name}"
        item["path"] = str(base / remote)
        inputs.append((local, remote))

    # 权重与 inputs 同样打包；远端 task.py 会重算 weights_sha，oracle 身份随之变化。
    if weights := rewritten.get("weights"):
        local = Path(weights)
        if not local.is_absolute():
            local = task_path.parent / local
        local = _inside_workspace(local)
        remote = f"weights/{local.name}"
        rewritten["weights"] = str(base / remote)
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
        exports = [f"export PYTHONPATH={shlex.quote(str(base / 'src'))}"]
        if SYSTEM_LIBSTDCXX:
            exports.append(f"export LD_PRELOAD={shlex.quote(SYSTEM_LIBSTDCXX)}")
        command = " && ".join(
            [
                f"cd {shlex.quote(str(base))}",
                *exports,
                f"{REMOTE_SOURCE_PYTHON} -m torchport {quoted}",
            ]
        )
        code, out, err = _run(client, command)
        sys.stdout.write(out)
        sys.stderr.write(err)
        return code
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
