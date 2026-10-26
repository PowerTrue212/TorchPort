"""Check every copied source-oracle file against the frozen oracle manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.oracle_dir / "oracle.json").read_text("utf-8"))
    files = {}
    for name, expected in manifest["files"].items():
        path = args.oracle_dir / name
        actual = sha256(path)
        files[name] = {"expected_sha256": expected, "actual_sha256": actual,
                       "bytes": path.stat().st_size, "matches": actual == expected}
    result = {"task_id": manifest["task_id"], "task_digest": manifest["task_digest"],
              "files": files, "passed": all(item["matches"] for item in files.values())}
    (args.oracle_dir / "copy-validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"],
                      "total_bytes": sum(item["bytes"] for item in files.values()),
                      "files": {name: item["matches"] for name, item in files.items()}}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
