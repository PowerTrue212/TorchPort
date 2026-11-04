"""Warm one JITTOR_HOME with the frozen FNO inference reference (no candidate code)."""
import sys
from pathlib import Path

import jittor as jt
import numpy as np

R = Path('/root/autodl-tmp/FNO-Benchmark-v1')
INF = R / 'inference-r1'
sys.path[:0] = [str(INF / 'reference-r1')]

from fno_reference import FNOReference  # noqa: E402

jt.flags.use_cuda = 1
model = FNOReference()
model.eval()
with np.load(INF / 'weights.npz') as archive:
    model.load_parameters({name: np.asarray(archive[name]) for name in archive.files})
x = jt.array(np.load(INF / 'inputs/0-x.npy'))
out = model(x)
print('output', out.shape, float(np.abs(out.numpy()).max()))
