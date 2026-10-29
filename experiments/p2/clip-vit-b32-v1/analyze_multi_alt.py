"""Pilot sensitivity check using the widest of two source alt captures.

This does not change TorchPort's frozen single-alt verification decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def errors(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64)).ravel()
    return float(np.max(difference)), float(np.quantile(difference, 0.99))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--fp64", type=Path, required=True)
    parser.add_argument("--alt", type=Path, required=True, nargs=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    failed = []
    with np.load(args.source) as source, np.load(args.target) as target, np.load(
        args.fp64
    ) as fp64, np.load(args.alt[0]) as alt_a, np.load(args.alt[1]) as alt_b:
        names = [name for name in source.files if name.startswith("output.")]
        for name in names:
            reference = source[name]
            truth = fp64[name]
            candidate_max, candidate_p99 = errors(target[name], truth)
            source_max, source_p99 = errors(reference, truth)
            alt_a_max, alt_a_p99 = errors(reference, alt_a[name])
            alt_b_max, alt_b_p99 = errors(reference, alt_b[name])
            noise_max = max(source_max, alt_a_max, alt_b_max)
            noise_p99 = max(source_p99, alt_a_p99, alt_b_p99)
            if candidate_max > 8.0 * noise_max + 1e-7 or (
                candidate_p99 > 8.0 * noise_p99 + 1e-7
            ):
                failed.append({
                    "name": name,
                    "candidate_max": candidate_max,
                    "candidate_p99": candidate_p99,
                    "noise_max": noise_max,
                    "noise_p99": noise_p99,
                })
    result = {
        "kind": "pilot-two-alt-sensitivity",
        "formal_decision": False,
        "compared_tensors": len(names),
        "failed_tensors": len(failed),
        "failed": failed,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "failed"}))


if __name__ == "__main__":
    main()
