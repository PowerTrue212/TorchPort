"""Compare two LightGCN source oracle outputs given their run directories."""
import sys
from pathlib import Path

import numpy as np

left = Path(sys.argv[1]) / 'outputs.npz'
right = Path(sys.argv[2]) / 'outputs.npz'
a = np.load(left)
b = np.load(right)
print('files', a.files, b.files)
for name in a.files:
    if name not in b.files:
        print('missing', name)
        continue
    x, y = np.asarray(a[name]), np.asarray(b[name])
    if x.shape != y.shape:
        print(name, 'shape differs', x.shape, y.shape)
        continue
    diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
    print(name, x.shape, 'bitwise_equal', np.array_equal(x, y),
          'max_abs', float(diff.max()) if diff.size else 0.0,
          'nonzero', int((diff > 0).sum()))
