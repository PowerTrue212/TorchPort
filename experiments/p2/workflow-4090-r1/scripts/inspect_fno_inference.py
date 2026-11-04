"""Inspect the frozen FNO inference materials before building the inference TaskSpec."""
import numpy as np
from pathlib import Path

root = Path('/root/autodl-tmp/FNO-Benchmark-v1/task-metrics-r2')
x = np.load(root / 'test-inputs.npy')
print('test-inputs', x.shape, x.dtype, float(x.min()), float(x.max()))
with np.load(root / 'weights.npz') as archive:
    print('weights keys', len(archive.files))
    for name in archive.files[:12]:
        print('  ', name, archive[name].shape, archive[name].dtype)
with np.load(root / 'test-predictions.npz') as archive:
    print('predictions keys', archive.files)
    for name in archive.files:
        print('  ', name, archive[name].shape, archive[name].dtype)
print((root / 'result.json').read_text())
