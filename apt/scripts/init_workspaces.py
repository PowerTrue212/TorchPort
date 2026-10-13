from __future__ import annotations

import argparse
import json
import shutil

from common import ROOT, load_config, method_ids, tree_digest, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create three identical APT migration workspaces")
    parser.add_argument("--method", choices=method_ids(), action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    source = ROOT / "source" / "apt"
    bootstrap_path = ROOT / "shared" / "provenance" / "bootstrap.json"
    if not source.is_dir() or not bootstrap_path.is_file():
        raise RuntimeError("run bootstrap.py first")
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    source_hash = tree_digest(source)
    if source_hash != bootstrap["source_tree_sha256"]:
        raise RuntimeError(f"source tree mismatch: {source_hash}")

    selected = args.method or list(method_ids())
    template = ROOT / "templates" / "submission.py"
    for method in selected:
        workspace = ROOT / "workspaces" / method
        repo = workspace / "repo"
        if workspace.exists():
            raise RuntimeError(f"refusing to overwrite experiment workspace: {workspace}")
        workspace.mkdir(parents=True)
        shutil.copytree(
            source,
            repo,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
        shutil.copy2(template, repo / "submission.py")
        write_json(
            workspace / "workspace_manifest.json",
            {
                "protocol_version": config["protocol_version"],
                "method": method,
                "initial_source_tree_sha256": source_hash,
                "initial_workspace_tree_sha256": tree_digest(repo),
                "source_archive_sha256": config["source"]["archive_sha256"],
                "dataset_archive_sha256": config["dataset"]["archive_sha256"],
                "rounds_file": f"results/{method}/rounds.jsonl",
            },
        )
        print(f"initialized {method}: {repo}")

    hashes = {method: tree_digest(ROOT / "workspaces" / method / "repo") for method in selected}
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"workspaces are not identical: {hashes}")
    print(f"initial workspace tree sha256: {next(iter(hashes.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

