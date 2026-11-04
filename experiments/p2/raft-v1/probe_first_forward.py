"""Compare all 12 RAFT step-zero flow predictions before backward."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/RAFT-Benchmark-v1")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path[:0] = [str(ROOT / "deps"), str(ROOT / "source/RAFT"), str(ROOT / "source/RAFT/core")]
from raft import RAFT  # noqa: E402
from source_adapter import Args, load_train_model  # noqa: E402

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
a = torch.from_numpy(np.load(ROOT / "inputs-r1/train-image1.npy")[0]).cuda()
b = torch.from_numpy(np.load(ROOT / "inputs-r1/train-image2.npy")[0]).cuda()
gt = torch.from_numpy(np.load(ROOT / "inputs-r1/train-flow.npy")[0]).cuda()
state = torch.load(ROOT / "checkpoints/raft-things.pth", map_location="cpu", weights_only=True)
state = {key.removeprefix("module."): value for key, value in state.items()}
official = RAFT(Args(small=False, mixed_precision=False, alternate_corr=False, dropout=0)).cuda().train()
official.load_state_dict(state, strict=True)
with torch.no_grad():
    original = [value.detach().cpu().numpy() for value in official(a, b, iters=12)]
custom = load_train_model().cuda().train()
custom.load_state_dict(state, strict=True)
with torch.no_grad():
    current = [value.detach().cpu().numpy() for value in custom(a, b, iters=12)]
flow_exact = [np.array_equal(x, y) for x, y in zip(original, current)]
mask = gt.square().sum(dim=1).sqrt() < 400
def loss(values):
    return sum(0.8 ** (11 - i) * (mask[:, None] * (torch.from_numpy(value).cuda() - gt).abs()).mean()
               for i, value in enumerate(values)).item()
result = {"status": "pilot_only", "all_flows_exact": all(flow_exact),
          "per_iteration_exact": flow_exact, "official_loss": loss(original), "custom_loss": loss(current)}
out = ROOT / "evidence/first-forward-r1.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
