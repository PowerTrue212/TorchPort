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
# 权重按内容哈希共享，避免每次 trace/verify 重传数百 MB。
WEIGHTS_CACHE = REMOTE_ROOT / "weights-cache"
SKIP_PARTS = {".git", ".torchport", "__pycache__", ".pytest_cache", "node_modules"}
TASK_COMMANDS = {"inspect", "run", "evaluate", "trace", "verify"}
COMMANDS = TASK_COMMANDS | {"localize"}
# diagnostic：只返回阶段、失败类别与聚合数值，不返回层名、张量名或首次分歧；
# localized：返回完整报告。完整报告只留在远端 evidence 下。
FEEDBACK = os.environ.get("TP_FEEDBACK", "localized")
LOCALIZING_COMMANDS = {"trace", "localize"}
# DSH 从 Agent 工作区调用本脚本，cwd 是工作区而不是 TorchPort 仓库；
# 打包路径校验和 torchport 源码位置都以仓库根为准。
PROJECT_ROOT = Path(os.environ.get("TP_PROJECT_ROOT") or Path(__file__).resolve().parents[2]).resolve()


def _inside_workspace(path: str | Path) -> Path:
    root = PROJECT_ROOT
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


def _local_weights(task_path: Path, task: dict[str, Any]) -> tuple[Path, str] | None:
    """返回 (本地权重路径, sha256)；任务没有权重时返回 None。"""
    weights = task.get("weights")
    if not weights:
        return None
    local = Path(weights)
    if not local.is_absolute():
        local = task_path.parent / local
    local = _inside_workspace(local)
    sha = hashlib.sha256()
    with local.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return local, sha.hexdigest()


def _cached_weights_path(local: Path, sha: str) -> PurePosixPath:
    # 目录名是内容哈希，文件名保留原名（load_checkpoint 按后缀选择读取方式）。
    return WEIGHTS_CACHE / sha / local.name


def _ensure_weights(client: Any, local: Path, sha: str) -> None:
    """远端缓存命中则跳过；未命中时上传到临时文件，核对 sha256 后再原子改名。"""
    remote = _cached_weights_path(local, sha)
    code, _, _ = _run(client, f"test -f {shlex.quote(str(remote))}", 60)
    if code == 0:
        return
    partial = f"{remote}.partial"
    code, _, err = _run(client, f"mkdir -p {shlex.quote(str(remote.parent))}", 60)
    if code:
        raise RuntimeError(err)
    sftp = client.open_sftp()
    try:
        sftp.put(str(local), partial)
    finally:
        sftp.close()
    code, out, err = _run(client, f"sha256sum {shlex.quote(partial)}", 600)
    if code or out.split()[:1] != [sha]:
        _run(client, f"rm -f {shlex.quote(partial)}", 60)
        raise RuntimeError(f"远端权重校验失败：{err or out}")
    code, _, err = _run(client, f"mv -f {shlex.quote(partial)} {shlex.quote(str(remote))}", 60)
    if code:
        raise RuntimeError(err)


def _build_bundle(task_path: Path, base: PurePosixPath) -> bytes:
    task: dict[str, Any] = json.loads(task_path.read_text("utf-8"))
    source = _inside_workspace(task["source"]["repo"])
    target = _inside_workspace(task["target"]["repo"])
    package_root = PROJECT_ROOT / "src" / "torchport"
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

    # 权重不进 bundle：按内容哈希放在远端共享缓存，由 _sync 保证存在；
    # 远端 task.py 仍会重算 weights_sha，oracle 身份不依赖缓存路径。
    if weights := _local_weights(task_path, rewritten):
        rewritten["weights"] = str(_cached_weights_path(*weights))

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
    task: dict[str, Any] = json.loads(task_path.read_text("utf-8"))
    if weights := _local_weights(task_path, task):
        _ensure_weights(client, *weights)
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


def diagnostic_view(report: dict[str, Any]) -> dict[str, Any]:
    """verify/evaluate 报告的 diagnostic 视图：阶段、失败类别、聚合数值、结构与权重状态。

    不含任何模块名、张量名、文件、行号、首次分歧或逐张量列表。
    """
    result = (report.get("results") or [report])[-1]
    evidence = result.get("evidence") or {}
    weights = evidence.get("weights") or {}
    e1 = report.get("e1") or {}
    metrics = e1.get("metrics") or {}
    if result.get("runnable") == "fail":
        stage = "runnable"
    elif result.get("structural") == "fail":
        stage = "structural"
    elif result.get("numerical") == "fail":
        stage = "numerical"
    elif report.get("passed"):
        stage = "passed"
    else:
        stage = "verification"
    return {
        "passed": bool(report.get("passed")),
        "stage": stage,
        "failure_code": report.get("failure_code"),
        "structural": result.get("structural", "not_run"),
        "weight_load": "pass" if weights.get("complete") else ("fail" if weights else "not_run"),
        "compared_tensors": metrics.get("compared_tensors"),
        "failed_tensors": metrics.get("failed_tensors"),
        "max_ratio": metrics.get("max_ratio"),
    }


def _filter(command: str, out: str) -> str:
    """diagnostic 条件下只把 diagnostic 视图交给 Agent；完整报告只留在远端 evidence 下。"""
    if FEEDBACK == "localized" or command not in {"verify", "evaluate"}:
        return out
    try:
        report = json.loads(out)
    except json.JSONDecodeError:
        return json.dumps({"passed": False, "stage": "verification", "failure_code": "UNPARSEABLE"})
    # 完整报告只留在远端 evidence/verify 下供事后评测，不在本机落盘（Agent 本机可读）。
    return json.dumps(diagnostic_view(report), ensure_ascii=False, indent=2)


def main(arguments: list[str] | None = None) -> int:
    arguments = list(arguments or sys.argv[1:])
    if not arguments or arguments[0] not in COMMANDS:
        raise SystemExit("仅支持 inspect/run/evaluate/trace/localize/verify")
    if FEEDBACK not in {"diagnostic", "localized"}:
        raise SystemExit(f"未知反馈条件：{FEEDBACK}")
    # run --mode trace 会产出逐层证据，与 trace 命令同属定位能力。
    mode = arguments[arguments.index("--mode") + 1] if "--mode" in arguments[:-1] else "normal"
    localizing = arguments[0] in LOCALIZING_COMMANDS or (arguments[0] == "run" and mode != "normal")
    if FEEDBACK == "diagnostic" and localizing:
        sys.stdout.write(json.dumps({"error": f"{arguments[0]} 在本实验条件下不可用"}, ensure_ascii=False))
        return 2
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
        sys.stdout.write(_filter(arguments[0], out))
        sys.stderr.write(err)
        return code
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
