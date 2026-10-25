"""Score a target's 100-image zero-shot logits against the frozen source sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(pilot: Path, target_path: Path, tau: float | None) -> dict:
    manifest = json.loads((pilot / "source_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha256(pilot / name) != expected:
            raise ValueError(f"Frozen pilot file changed: {name}")
    labels = np.load(pilot / "labels.npy", allow_pickle=False)
    source = np.load(pilot / "source_logits.npy", allow_pickle=False)
    if target_path.suffix.lower() == ".npz":
        with np.load(target_path, allow_pickle=False) as archive:
            target = np.asarray(archive["output"])
    else:
        target = np.load(target_path, allow_pickle=False)
    if source.shape != (3, 100, 10) or target.shape != (100, 10):
        raise ValueError("Expected source (3,100,10) and target (100,10) logits")
    if not np.isfinite(target).all():
        raise ValueError("Target logits contain non-finite values")
    source_predictions = source[0].argmax(axis=1)
    target_predictions = target.argmax(axis=1)
    source_accuracy = float((source_predictions == labels).mean())
    target_accuracy = float((target_predictions == labels).mean())
    gap = abs(target_accuracy - source_accuracy)
    if tau is not None and tau <= 0:
        raise ValueError("tau must be positive")
    return {
        "status": "pilot_only",
        "source_top1_accuracy": source_accuracy,
        "target_top1_accuracy": target_accuracy,
        "absolute_accuracy_gap": gap,
        "prediction_match_rate": float((source_predictions == target_predictions).mean()),
        "tau": tau,
        "P_t": None if tau is None else 100.0 / max(1.0, gap / tau),
        "task_success": None if tau is None else gap <= tau,
        "target_logits_sha256": sha256(target_path),
        "source_manifest_sha256": sha256(pilot / "source_manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--target-logits", type=Path, required=True)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.pilot, args.target_logits, args.tau)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
