"""Print keys/shapes of the frozen FNO oracle state and the task-metrics weights."""
import numpy as np
from pathlib import Path

root = Path('/root/autodl-tmp/FNO-Benchmark-v1')
paths = [
    root / 'oracle-bundle-r3/evidence/oracle/f3d97351178dd9ff/state.npz',
    root / 'oracle-bundle-r3/evidence/oracle/f3d97351178dd9ff/outputs.npz',
    root / 'task-metrics-r2/weights.npz',
]
for path in paths:
    print('==', path, path.is_file())
    if not path.is_file():
        continue
    with np.load(path) as archive:
        print('  keys', len(archive.files))
        for name in archive.files[:6]:
            value = archive[name]
            print('   ', name, value.shape, value.dtype)
