"""Compare two frozen FNO source pilot runs, including all model tensors."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1])
first = root / "source-pilot-r2"
second = root / "source-pilot-r3"
result = {"status": "pilot_only", "arrays": {}, "states": {}}

for filename in ("inputs.npz", "outputs.npz"):
    with np.load(first / filename) as a, np.load(second / filename) as b:
        assert a.files == b.files
        result["arrays"][filename] = {
            name: {
                "shape": list(a[name].shape),
                "equal": bool(np.array_equal(a[name], b[name])),
                "max_abs": float(np.max(np.abs(a[name] - b[name]))),
            }
            for name in a.files
        }

for filename in ("initial.pt", "post.pt"):
    a = torch.load(first / filename, map_location="cpu", weights_only=False)
    b = torch.load(second / filename, map_location="cpu", weights_only=False)
    assert a.keys() == b.keys()
    tensors = [name for name, value in a.items() if isinstance(value, torch.Tensor)]
    differences = {
        name: float(torch.max(torch.abs(a[name] - b[name])).item())
        for name in tensors
        if not torch.equal(a[name], b[name])
    }
    result["states"][filename] = {
        "tensor_count": len(tensors),
        "differing_tensors": differences,
        "metadata_equal": a.get("_metadata") == b.get("_metadata"),
    }

print(json.dumps(result, indent=2))
