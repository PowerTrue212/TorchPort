"""Score frozen STS-B embeddings without filling missing target results with zero."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def similarities(path: Path, pairs: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "output" not in archive.files:
            raise ValueError(f"Missing output tensor in {path}")
        embeddings = np.asarray(archive["output"], dtype=np.float64)
    if embeddings.shape != (2 * pairs, 768) or not np.isfinite(embeddings).all():
        raise ValueError(f"Wrong shape or non-finite embeddings: {path}")
    first, second = embeddings[:pairs], embeddings[pairs:]
    norms = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    if np.any(norms == 0):
        raise ValueError(f"Zero-norm sentence embedding: {path}")
    return np.sum(first * second, axis=1) / norms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--source-embeddings", type=Path, required=True)
    parser.add_argument("--target-embeddings", type=Path)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    labels = np.load(args.labels, allow_pickle=False)
    if labels.ndim != 1 or not np.isfinite(labels).all():
        raise ValueError("Labels must be a finite one-dimensional array")
    if args.tau is not None and args.tau <= 0:
        raise ValueError("tau must be positive")
    scores = {"source": similarities(args.source_embeddings, len(labels))}
    if args.target_embeddings is not None:
        scores["target"] = similarities(args.target_embeddings, len(labels))
    correlations = {name: float(spearmanr(labels, values).correlation)
                    for name, values in scores.items()}
    if not np.isfinite(list(correlations.values())).all():
        raise ValueError("Spearman correlation is non-finite")
    gap = (abs(correlations["target"] - correlations["source"])
           if "target" in correlations else None)
    ratio = gap / args.tau if gap is not None and args.tau is not None else None
    result = {
        "pairs": len(labels),
        "source_spearman": correlations["source"],
        "target_spearman": correlations.get("target"),
        "absolute_gap": gap,
        "tau": args.tau,
        "P_t": 100.0 / max(1.0, ratio) if ratio is not None else None,
        "task_success": ratio <= 1.0 if ratio is not None else None,
        "source_embeddings_sha256": sha256(args.source_embeddings),
        "target_embeddings_sha256": (sha256(args.target_embeddings)
                                     if args.target_embeddings else None),
        "labels_sha256": sha256(args.labels),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "cosine-scores.npz", **scores)
    (args.output_dir / "score.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
