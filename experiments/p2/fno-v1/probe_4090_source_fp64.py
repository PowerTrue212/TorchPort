"""Check whether the frozen FNO source path supports a true FP64 train step."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "source"))
from torchport_train_adapter import build_optimizer, load_train_model, train_step  # noqa: E402

torch.manual_seed(20261030)
torch.cuda.manual_seed_all(20261030)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

out = {}
for dtype in (torch.float32, torch.float64):
    model = load_train_model().cuda()
    if dtype == torch.float64:
        model.double()
    weights = np.load(root / "train-inputs-4090-r2/initial-weights.npz")
    model.load_state_dict({key: torch.tensor(weights[key], device="cuda", dtype=model.state_dict()[key].dtype)
                           for key in weights.files})
    optimizer = build_optimizer(model)
    x = torch.from_numpy(np.load(root / "train-inputs-4090-r2/x.npy")[0]).to("cuda", dtype=dtype)
    y = torch.from_numpy(np.load(root / "train-inputs-4090-r2/y.npy")[0]).to("cuda", dtype=dtype)
    loss = train_step(model, optimizer, x, y)
    out[str(dtype)] = {"loss": float(loss.item()), "finite": bool(torch.isfinite(loss))}
print(json.dumps(out, indent=2))
