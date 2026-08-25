from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

from common import ROOT, load_config, method_ids, verify_manifest, write_json


def source_tree_sha256(root: Path) -> str:
    checksum = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        checksum.update(len(relative).to_bytes(8, "big"))
        checksum.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(chunk)
    return checksum.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create three identical RAFT workspaces")
    parser.add_argument("--method", choices=method_ids(), action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    reference = verify_manifest("reference")
    source = ROOT / "source" / "princeton-vl-raft"
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != config["model"]["commit"]:
        raise RuntimeError(f"source commit mismatch: {head}")
    tree_hash = source_tree_sha256(source)
    template = ROOT / "templates" / "submission.py"
    selected = args.method or list(method_ids())

    for method in selected:
        workspace = ROOT / "workspaces" / method
        repo = workspace / "repo"
        existing = [path for path in workspace.iterdir()] if workspace.exists() else []
        if existing == [repo] and repo.is_dir() and not any(repo.iterdir()):
            repo.rmdir()
            existing = []
        if existing:
            raise RuntimeError(
                f"workspace already contains experiment state; refusing overwrite: {workspace}"
            )
        workspace.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source,
            repo,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
        shutil.copy2(template, repo / "submission.py")
        write_json(
            workspace / "workspace_manifest.json",
            {
                "protocol_version": config["protocol_version"],
                "method": method,
                "raft_commit": head,
                "initial_source_tree_sha256": tree_hash,
                "checkpoint_sha256": config["model"]["checkpoint_sha256"],
                "prepared_files": reference["files"],
                "rounds_file": f"results/{method}/rounds.jsonl",
            },
        )
        (ROOT / "results" / method).mkdir(parents=True, exist_ok=True)
        print(f"initialized {method}: {repo}")

    hashes = {
        method: source_tree_sha256(ROOT / "workspaces" / method / "repo") for method in selected
    }
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"workspaces are not identical: {hashes}")
    print(f"initial workspace tree sha256: {next(iter(hashes.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
