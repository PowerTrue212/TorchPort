"""Repository-level candidate delivery and reproducible snapshots."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from torchport.contracts import SKIP_DIRS, digest, file_hash
from torchport.runtime import write_json

DELIVERY_SKIP_DIRS = set(SKIP_DIRS) | {".git"}
DELIVERY_SKIP_ROOT_DIRS = {".scratch"}
SENSITIVE_NAMES = {".env", "id_dsa", "id_ecdsa", "id_ed25519", "id_rsa"}


def _walk(root: Path):
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        child_directories = []
        for name in sorted(names):
            if name in DELIVERY_SKIP_DIRS or (base == root and name in DELIVERY_SKIP_ROOT_DIRS):
                continue
            path = base / name
            if path.is_symlink():
                yield path, path.relative_to(root)
            else:
                child_directories.append(name)
        names[:] = child_directories
        for name in sorted(files):
            path = base / name
            relative = path.relative_to(root)
            yield path, relative


def repository_manifest(root: str | Path) -> dict[str, Any]:
    """Describe every delivered repository file without depending on its absolute path."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = []
    for path, relative in _walk(root):
        lower_name = path.name.lower()
        if lower_name in SENSITIVE_NAMES or lower_name.startswith(".env."):
            raise ValueError(f"candidate 不能包含凭据文件：{relative.as_posix()}")
        if path.is_symlink():
            target = os.readlink(path)
            resolved = path.resolve(strict=False)
            if resolved != root and root not in resolved.parents:
                raise ValueError(f"candidate 符号链接不能指向仓库外：{relative.as_posix()}")
            files.append(
                {
                    "path": relative.as_posix(),
                    "kind": "symlink",
                    "target": target,
                    "sha256": digest({"target": target}),
                }
            )
            continue
        if not path.is_file():
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "kind": "file",
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
        )
    return {
        "schema_version": 1,
        "kind": "torchport.repository-candidate",
        "files": files,
        "file_count": len(files),
        "content_digest": digest(files),
    }


def _git(
    root: Path,
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
        env=env,
    )


def initialize_git_workspace(root: str | Path) -> bool:
    """Create an isolated baseline commit for a newly materialized candidate workspace."""
    root = Path(root).resolve()
    if (root / ".git").exists():
        return False
    commands = (
        ("init", "--quiet"),
        ("add", "-A"),
        (
            "-c",
            "user.name=TorchPort",
            "-c",
            "user.email=torchport@invalid",
            "commit",
            "--quiet",
            "-m",
            "Initial repository candidate",
        ),
    )
    for command in commands:
        completed = _git(root, *command)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "git workspace initialization failed")
    return True


def _git_evidence(root: Path, output: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        return {"available": False, "reason": "candidate workspace has no Git baseline"}
    scratch_exclude = ":(exclude).scratch"
    status = _git(root, "status", "--short", "--untracked-files=all", "--", ".", scratch_exclude)
    index_path = output / ".candidate-index"
    git_env = os.environ.copy()
    git_env["GIT_INDEX_FILE"] = str(index_path)
    read_tree = _git(root, "read-tree", "HEAD", env=git_env)
    add = _git(root, "add", "-A", "--", ".", scratch_exclude, env=git_env)
    diff = _git(
        root,
        "diff",
        "--cached",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        "--",
        ".",
        scratch_exclude,
        env=git_env,
    )
    index_path.unlink(missing_ok=True)
    if any(item.returncode != 0 for item in (status, read_tree, add, diff)):
        return {
            "available": False,
            "reason": (
                status.stderr
                or read_tree.stderr
                or add.stderr
                or diff.stderr
                or "git evidence failed"
            ).strip(),
        }
    status_path = output / "git-status.txt"
    diff_path = output / "candidate.diff"
    status_path.write_text(status.stdout, encoding="utf-8")
    diff_path.write_text(diff.stdout, encoding="utf-8")
    return {
        "available": True,
        "status": status_path.name,
        "status_sha256": file_hash(status_path),
        "diff": diff_path.name,
        "diff_sha256": file_hash(diff_path),
    }


def finalize_candidate(
    root: str | Path,
    output: str | Path,
    *,
    required_entry: str | None = None,
) -> dict[str, Any]:
    """Freeze a clean repository snapshot and its review evidence before official verify."""
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("candidate delivery 目录必须位于候选仓库之外")
    output.mkdir(parents=True, exist_ok=True)
    if required_entry:
        entry = (root / required_entry).resolve()
        if entry.parent != root and root not in entry.parents:
            raise ValueError("candidate entry 必须位于仓库内")
        if not entry.is_file():
            raise ValueError(f"candidate 缺少标准入口：{required_entry}")

    before = repository_manifest(root)
    if required_entry:
        other_files = [item for item in before["files"] if item["path"] != required_entry]
        if not other_files:
            raise ValueError("candidate 必须交付完整仓库，不能只有入口文件")

    snapshot = output / "repository"
    if snapshot.exists():
        raise FileExistsError(f"candidate snapshot 已存在：{snapshot}")
    shutil.copytree(
        root,
        snapshot,
        symlinks=True,
        ignore=lambda _directory, names: [
            name
            for name in names
            if name in DELIVERY_SKIP_DIRS
            or (Path(_directory) == root and name in DELIVERY_SKIP_ROOT_DIRS)
        ],
    )
    after = repository_manifest(snapshot)
    if before["content_digest"] != after["content_digest"]:
        raise RuntimeError("candidate snapshot 内容与工作区不一致")

    git = _git_evidence(root, output)
    report = {
        **after,
        "required_entry": required_entry,
        "snapshot": str(snapshot),
        "git": git,
    }
    manifest_path = output / "candidate-manifest.json"
    write_json(manifest_path, report)
    report["manifest"] = str(manifest_path)
    report["manifest_sha256"] = file_hash(manifest_path)
    return report
