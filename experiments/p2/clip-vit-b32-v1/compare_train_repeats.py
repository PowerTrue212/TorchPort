"""Summarize differences between two raw CLIP training captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    with np.load(args.first, allow_pickle=False) as first, np.load(
        args.second, allow_pickle=False
    ) as second:
        if set(first.files) != set(second.files):
            raise ValueError("Target captures have different tensor names")
        for name in sorted(first.files):
            left = first[name]
            right = second[name]
            if left.shape != right.shape:
                raise ValueError(f"Shape changed: {name}")
            count = int(np.count_nonzero(left != right))
            if count:
                delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
                summaries.append({
                    "name": name,
                    "different_elements": count,
                    "max_abs_delta": float(np.max(delta)),
                })
        result = {
            "schema_version": 1,
            "kind": "clip-train-repeat-comparison",
            "compared_tensors": len(first.files),
            "different_tensors": len(summaries),
            "different_elements": sum(item["different_elements"] for item in summaries),
            "max_abs_delta": max((item["max_abs_delta"] for item in summaries), default=0.0),
            "first_differences": summaries[:20],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "first_differences"}))


if __name__ == "__main__":
    main()
