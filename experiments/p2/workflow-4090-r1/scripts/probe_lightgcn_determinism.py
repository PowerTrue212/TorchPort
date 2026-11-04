"""Probe whether the LightGCN source path can be made bitwise reproducible."""
import os
import sys

R = '/root/autodl-tmp/LightGCN-Benchmark-v1'
os.environ['TP_LIGHTGCN_DEPS'] = f'{R}/deps'
os.environ.setdefault('HOME', f'{R}/home')
sys.path[:0] = [f'{R}/inference-r1/source/LightGCN-PyTorch-pilot-r2']

import numpy as np  # noqa: E402
import torch  # noqa: E402

mode = sys.argv[1] if len(sys.argv) > 1 else 'plain'
if mode == 'det':
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

from torchport_inference_adapter_r2 import load_inference_model  # noqa: E402

model = load_inference_model()
model = model.cuda().eval()
with np.load(f'{R}/inference-r1/weights.npz') as archive:
    state = {name: torch.from_numpy(np.asarray(archive[name])) for name in archive.files}
model.load_state_dict(state, strict=True)
users = torch.from_numpy(np.load(f'{R}/inference-r1/inputs/0-users.npy')).long().cuda()
with torch.no_grad():
    first = model(users).cpu().numpy().copy()
    second = model(users).cpu().numpy().copy()
print(mode, 'shape', first.shape, 'bitwise_equal', np.array_equal(first, second),
      'max_abs', float(np.abs(first - second).max()), 'graph_device', model.Graph.device)
