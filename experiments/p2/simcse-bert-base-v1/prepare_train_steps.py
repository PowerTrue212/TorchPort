"""Freeze two identical SimCSE training batches for the Harness task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STEPS = 2
NAMES = ("input_ids", "attention_mask", "token_type_ids")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_manifest = args.batch_dir / "manifest.json"
    source = json.loads(source_manifest.read_text("utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in NAMES:
        source_path = args.batch_dir / f"{name}.npy"
        if sha256(source_path) != source["files"][name]["sha256"]:
            raise ValueError(f"Frozen batch hash differs: {name}")
        batch = np.load(source_path, allow_pickle=False)
        if batch.shape != (8, 2, 32) or batch.dtype != np.int64:
            raise ValueError(f"Frozen batch shape or dtype differs: {name}")
        steps = np.stack((batch, batch), axis=0)
        path = args.output_dir / f"{name}.npy"
        np.save(path, steps)
        files[name] = {"shape": list(steps.shape), "dtype": str(steps.dtype),
                       "sha256": sha256(path)}
    manifest = {"schema_version": 1, "source_batch_manifest_sha256": sha256(source_manifest),
                "step_count": STEPS, "step_rule": "repeat the same fixed batch",
                "files": files}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"shape": files["input_ids"]["shape"],
                      "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
