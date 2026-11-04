"""Report first differences between two independent source training captures."""
import glob
import sys
from pathlib import Path

import numpy as np

series = sys.argv[1]
root = Path(f"/root/autodl-tmp/RAFT-Benchmark-v1/oracle-bundle-train-three-r1-{series}/evidence/runs")
paths = sorted(glob.glob(str(root / "oracle-*" / "source-*" / "outputs.npz")))
assert len(paths) == 2, paths
with np.load(paths[0]) as a, np.load(paths[1]) as b:
    assert set(a.files) == set(b.files)
    changed = []
    for key in a.files:
        x, y = a[key], b[key]
        if not np.array_equal(x, y):
            changed.append((key, float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))))
    print("tensors", len(a.files), "different", len(changed))
    for item in changed[:25]:
        print(item)
    for key in sorted(k for k in a.files if "loss" in k or "lr" in k):
        print(key, a[key], b[key])
