"""构建远端工作包并同步冻结任务。"""

from __future__ import annotations

import hashlib
import io
import json
import shlex
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from . import transport
from .config import RemoteSettings

SKIP_PARTS = {".git", ".torchport", "__pycache__", ".pytest_cache", "node_modules"}


def inside_workspace(path: str | Path, settings: RemoteSettings) -> Path:
    value = Path(path)
    value = value.resolve() if value.is_absolute() else (settings.project_root / value).resolve()
    for root in (settings.project_root, *settings.extra_roots):
        if value == root or root in value.parents:
            return value
    raise PermissionError(f"路径不在 TorchPort 工作区内：{value}")


def remote_base(task_path: Path, settings: RemoteSettings) -> PurePosixPath:
    digest = hashlib.sha256(str(task_path).encode()).hexdigest()[:12]
    return settings.root / digest


def report_base(report: str, settings: RemoteSettings) -> PurePosixPath:
    path = PurePosixPath(report)
    try:
        relative = path.relative_to(settings.root)
    except ValueError as exc:
        raise PermissionError("localize 只接受本次远端实验生成的报告") from exc
    if len(relative.parts) < 2:
        raise PermissionError("localize 报告路径缺少任务目录")
    return settings.root / relative.parts[0]


def _add_tree(package: tarfile.TarFile, path: Path, arcname: str) -> None:
    def allowed(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        return None if any(part in SKIP_PARTS or part.endswith(".pyc") for part in parts) else info

    package.add(path, arcname=arcname, recursive=True, filter=allowed)


def local_weights(
    task_path: Path, task: dict[str, Any], settings: RemoteSettings
) -> tuple[Path, str] | None:
    weights = task.get("weights")
    if not weights:
        return None
    local = Path(weights)
    if not local.is_absolute():
        local = task_path.parent / local
    local = inside_workspace(local, settings)
    sha = hashlib.sha256()
    with local.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return local, sha.hexdigest()


def cached_weights_path(local: Path, sha: str, settings: RemoteSettings) -> PurePosixPath:
    return settings.weights_cache / sha / local.name


def build_bundle(task_path: Path, base: PurePosixPath, settings: RemoteSettings) -> bytes:
    task: dict[str, Any] = json.loads(task_path.read_text("utf-8"))
    source = inside_workspace(task["source"]["repo"], settings)
    target = inside_workspace(task["target"]["repo"], settings)
    package_root = settings.project_root / "src" / "torchport"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)

    rewritten = json.loads(json.dumps(task))
    rewritten["source"].update(repo=str(base / "source"), python=settings.source_python)
    rewritten["target"].update(repo=str(base / "target"), python=settings.target_python)
    target_env = dict(rewritten["target"].get("env") or {})
    target_env.update({"PATH": settings.target_path, "use_cutlass": "0"})
    for name, value in (("LD_PRELOAD", settings.system_libstdcxx), ("python_config_path", settings.python_config)):
        if value:
            target_env[name] = value
        else:
            target_env.pop(name, None)
    for name, value in settings.target_env_defaults.items():
        if value:
            target_env.setdefault(name, value)
    rewritten["target"]["env"] = target_env

    inputs: list[tuple[Path, str]] = []
    for index, item in enumerate(rewritten.get("inputs") or []):
        local = inside_workspace(item["path"], settings)
        remote = f"inputs/{index}-{local.name}"
        item["path"] = str(base / remote)
        inputs.append((local, remote))
    if weights := local_weights(task_path, rewritten, settings):
        rewritten["weights"] = str(cached_weights_path(*weights, settings))

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


def sync(client: Any, task_path: Path, base: PurePosixPath, settings: RemoteSettings) -> None:
    task: dict[str, Any] = json.loads(task_path.read_text("utf-8"))
    if weights := local_weights(task_path, task, settings):
        local, sha = weights
        transport.ensure_weights(client, local, sha, cached_weights_path(local, sha, settings))
    archive = build_bundle(task_path, base, settings)
    remote_archive = str(base) + ".tar.gz"
    code, _, err = transport.run(client, f"mkdir -p {shlex.quote(str(settings.root))}", 60)
    if code:
        raise RuntimeError(err)
    sftp = client.open_sftp()
    try:
        with sftp.file(remote_archive, "wb") as output:
            output.write(archive)
    finally:
        sftp.close()
    replace = " ".join(shlex.quote(str(base / name)) for name in ("source", "target", "src", "inputs", "task.json"))
    command = (
        f"mkdir -p {shlex.quote(str(base))} && rm -rf {replace} && "
        f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(str(base))} && "
        f"rm -f {shlex.quote(remote_archive)}"
    )
    code, _, err = transport.run(client, command, 120)
    if code:
        raise RuntimeError(err)
