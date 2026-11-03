"""Freeze ten bundled Darcy samples and the source pilot's initial FNO state."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=False)
data_path = root / "source/neuralop/data/datasets/data/darcy_train_16.pt"
dataset = torch.load(data_path, map_location="cpu", weights_only=False)
for key in ("x", "y"):
    array = dataset[key][:10].unsqueeze(1).to(torch.float32).numpy()
    np.save(out / f"{key}.npy", array.reshape(5, 2, 1, 16, 16))

initial = torch.load(
    root / "source-pilot-4090-r1/initial.pt",
    map_location="cpu", weights_only=False,
)
weights = {name: value.detach().numpy() for name, value in initial.items()
           if isinstance(value, torch.Tensor)}
np.savez_compressed(out / "initial-weights.npz", **weights)
manifest = {
    "status": "pilot_only",
    "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    "sample_indices": list(range(10)),
    "batch_indices": [[2 * i, 2 * i + 1] for i in range(5)],
    "initial_state_tensor_count": len(weights),
    "initial_state_elements": sum(value.size for value in weights.values()),
    "files": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
              for name in ("x.npy", "y.npy", "initial-weights.npz")},
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
