"""Measure the FP32 source error against a frozen FP64 source run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def error(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape:
        raise ValueError(f"Tensor shapes differ: {left.shape} and {right.shape}")
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"shape": list(left.shape), "max_abs": float(difference.max()),
            "p99_abs": float(np.percentile(difference, 99))}


def compare_archives(left_path: Path, right_path: Path) -> dict:
    with np.load(left_path, allow_pickle=False) as left:
        with np.load(right_path, allow_pickle=False) as right:
            if left.files != right.files:
                raise ValueError("Tensor archive keys differ")
            return {name: error(left[name], right[name]) for name in left.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-dir", type=Path, required=True)
    parser.add_argument("--fp64-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left = args.fp32_dir
    right = args.fp64_dir
    losses = error(np.load(left / "losses.npy", allow_pickle=False),
                   np.load(right / "losses.npy", allow_pickle=False))
    logits = compare_archives(left / "logits.npz", right / "logits.npz")
    tensors = compare_archives(left / "selected_tensors.npz",
                               right / "selected_tensors.npz")
    with np.load(left / "selected_tensors.npz", allow_pickle=False) as a:
        with np.load(right / "selected_tensors.npz", allow_pickle=False) as b:
            updates = {}
            for name in a.files:
                if not name.startswith("initial.param."):
                    continue
                parameter = name.removeprefix("initial.param.")
                for step in (0, 1):
                    after = f"step_{step}.param.{parameter}"
                    updates[f"step_{step}.{parameter}"] = error(
                        a[after].astype(np.float64) - a[name].astype(np.float64),
                        b[after].astype(np.float64) - b[name].astype(np.float64),
                    )
    result = {"losses": losses, "logits": logits,
              "selected_tensors": tensors, "updates_from_initial": updates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"loss_max_abs": losses["max_abs"],
                      "logit_max_abs": max(item["max_abs"] for item in logits.values()),
                      "selected_gradient_max_abs": max(
                          item["max_abs"] for key, item in tensors.items()
                          if ".grad." in key),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
