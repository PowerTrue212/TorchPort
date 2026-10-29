"""Offline CLIP pilot: rejudge saved captures with a fixed reference repeat span.

The two r1 captures define the peer baseline.  Later captures are holdouts;
this diagnostic does not change their original hidden verify decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def error(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    difference = np.abs(np.asarray(left, np.float64) - np.asarray(right, np.float64)).ravel()
    return float(difference.max()), float(np.quantile(difference, 0.99))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fp64", type=Path, required=True)
    parser.add_argument("--alt", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, nargs=2, required=True)
    parser.add_argument("--holdout", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--floor", type=float, default=1e-7)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [args.source, args.fp64, args.alt, *args.calibration, *args.holdout]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    archives = [np.load(path, allow_pickle=False) for path in paths]
    summaries = [{"path": str(path), "sha256": digest(path), "failed": 0,
                  "max_threshold_ratio": 0.0, "worst": []} for path in paths[3:]]
    try:
        source, fp64, alt = archives[:3]
        first, second = archives[3:5]
        names = [name for name in source.files if name.startswith("output.")]
        for name in names:
            arrays = [archive[name] for archive in archives]
            if any(array.shape != arrays[0].shape for array in arrays):
                raise ValueError(f"shape differs: {name}")
            if any(not np.all(np.isfinite(array)) for array in arrays):
                raise ValueError(f"nonfinite: {name}")
            values = [array[:2] if name.startswith("output.ustat.") else array for array in arrays]
            s, f, a, peer_a, peer_b = values[:5]
            if name.startswith("output.ustat.") and arrays[0].size != 3:
                raise ValueError(f"ustat shape differs: {name}")
            sf, sa, peers = error(s, f), error(s, a), error(peer_a, peer_b)
            noise = [max(sf[i], sa[i], peers[i]) for i in range(2)]
            limits = [args.k * value + args.floor for value in noise]
            for summary, value in zip(summaries, values[3:], strict=True):
                candidate = error(value, f)
                ratio = max(candidate[i] / limits[i] for i in range(2))
                summary["max_threshold_ratio"] = max(summary["max_threshold_ratio"], ratio)
                if ratio > 1:
                    summary["failed"] += 1
                summary["worst"].append({"name": name, "threshold_ratio": ratio,
                                          "candidate_error": candidate, "noise": noise})
        for summary in summaries:
            summary["worst"] = sorted(summary["worst"], key=lambda row: -row["threshold_ratio"])[:10]
            summary["passed"] = summary["failed"] == 0
    finally:
        for archive in archives:
            archive.close()
    result = {"kind": "pilot-fixed-reference-repeat-sensitivity", "formal_decision": False,
              "rule": "drop ustat sum; max(source fp64, fixed alt, r1 peer diameter)",
              "k": args.k, "floor": args.floor, "compared_tensors": len(names),
              "source": {"path": str(args.source), "sha256": digest(args.source)},
              "fp64": {"path": str(args.fp64), "sha256": digest(args.fp64)},
              "alt": {"path": str(args.alt), "sha256": digest(args.alt)},
              "calibration": summaries[:2], "holdout": summaries[2:]}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"compared_tensors": len(names), "runs": [
        {key: value for key, value in summary.items() if key not in {"worst", "sha256"}}
        for summary in summaries]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
