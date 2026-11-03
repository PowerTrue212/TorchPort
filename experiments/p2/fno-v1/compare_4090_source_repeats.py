"""Compare two RTX 4090 source pilot runs byte for byte at tensor level."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1])
a_dir = root / "source-pilot-4090-r1"
b_dir = root / "source-pilot-4090-r2"
result = {"status": "pilot_only", "arrays": {}, "states": {}}

for filename in ("inputs.npz", "outputs.npz"):
    with np.load(a_dir / filename) as a, np.load(b_dir / filename) as b:
        assert a.files == b.files
        result["arrays"][filename] = {
            key: {"equal": bool(np.array_equal(a[key], b[key])),
                  "max_abs": float(np.max(np.abs(a[key] - b[key])))}
            for key in a.files
        }

for filename in ("initial.pt", "post.pt"):
    a = torch.load(a_dir / filename, map_location="cpu", weights_only=False)
    b = torch.load(b_dir / filename, map_location="cpu", weights_only=False)
    assert a.keys() == b.keys()
    names = [key for key in a if isinstance(a[key], torch.Tensor)]
    differing = [key for key in names if not torch.equal(a[key], b[key])]
    result["states"][filename] = {
        "tensor_count": len(names), "differing_tensors": differing
    }

print(json.dumps(result, indent=2))
