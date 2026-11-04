"""Compare official CUDA grid sampling to deterministic gather pilot outputs."""
import glob
from pathlib import Path

import numpy as np

root = Path("/root/autodl-tmp/RAFT-Benchmark-v1")
def first(series):
    matches = glob.glob(str(root / f"oracle-bundle-train-three-r1-{series}/evidence/runs/oracle-0/source-*/outputs.npz"))
    assert len(matches) == 1, matches
    return matches[0]

with np.load(first("r3")) as original, np.load(first("r4")) as deterministic:
    assert set(original.files) == set(deterministic.files)
    diffs = {}
    for key in original.files:
        a, b = original[key], deterministic[key]
        if not np.array_equal(a, b):
            diffs[key] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    print("tensors", len(original.files), "different", len(diffs))
    for key in sorted(k for k in original.files if "loss" in k or "lr" in k):
        print(key, original[key], deterministic[key])
    print("maximum absolute per tensor", max(diffs.values(), default=0.0))
    print("largest", sorted(diffs.items(), key=lambda item: -item[1])[:10])
