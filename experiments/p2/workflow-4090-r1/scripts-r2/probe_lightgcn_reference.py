"""Warm one JITTOR_HOME with the frozen LightGCN inference reference (no candidate code)."""
import sys
from pathlib import Path

import jittor as jt
import numpy as np

R = Path('/root/autodl-tmp/LightGCN-Benchmark-v1')
INF = R / 'inference-r1'
sys.path[:0] = [str(INF / 'reference-r2')]

from evaluation_entry import load_inference_model  # noqa: E402

jt.flags.use_cuda = 1
model = load_inference_model()
model.eval()
users = jt.array(np.load(INF / 'inputs/0-users.npy'))
with np.load(INF / 'weights.npz') as archive:
    model.load_parameters({name: np.asarray(archive[name]) for name in archive.files})
out = model(users)
print('output', out.shape, float(np.abs(out.numpy()).max()))
