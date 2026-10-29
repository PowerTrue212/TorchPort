"""Freeze per-tensor target repeat noise from two controller-held references.

The output is a judge-only pilot artifact. Do not place it in an Agent workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_a", type=Path)
    parser.add_argument("target_b", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task-digest", required=True)
    args = parser.parse_args()
    manifest = args.output.with_suffix(".json")
    if args.output.exists() or manifest.exists():
        raise FileExistsError(args.output)
    with (
        np.load(args.target_a, allow_pickle=False) as first,
        np.load(args.target_b, allow_pickle=False) as second,
    ):
        names = [name for name in first.files if name.startswith("output.")]
        if set(names) != {name for name in second.files if name.startswith("output.")}:
            raise ValueError("参考 target 输出集合不同")
        baseline = {}
        for name in names:
            a, b = first[name], second[name]
            if a.shape != b.shape or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                raise ValueError(f"参考 target 张量无效：{name}")
            subset = slice(0, 2) if name.startswith("output.ustat.") else slice(None)
            diff = np.abs(np.asarray(a[subset], np.float64) - np.asarray(b[subset], np.float64)).ravel()
            baseline[name] = np.asarray([diff.max(), np.quantile(diff, 0.99)], dtype=np.float64)
    np.savez_compressed(args.output, **baseline)
    record = {
        "profile": "train-repeat-envelope-v1-pilot",
        "task_digest": args.task_digest,
        "references": [
            {"path": str(path), "sha256": file_hash(path)}
            for path in (args.target_a, args.target_b)
        ],
        "tensors": len(baseline),
        "drop_ustat_sum": True,
        "baseline_path": str(args.output),
        "baseline_sha256": file_hash(args.output),
    }
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"tensors": record["tensors"], "sha256": record["baseline_sha256"]}))


if __name__ == "__main__":
    main()
