"""Diagnose the LightGCN reference graph representation and computer() shapes."""
import sys
from pathlib import Path

import jittor as jt
import numpy as np

R = Path('/root/autodl-tmp/LightGCN-Benchmark-v1')
INF = R / 'inference-r1'
sys.path[:0] = [str(INF / 'reference-r2')]

jt.flags.use_cuda = 1
with np.load(INF / 'reference-r2' / 'graph-r1.npz', allow_pickle=False) as graph:
    print('graph keys', graph.files)
    for name in graph.files:
        value = graph[name]
        print(' ', name, value.shape, value.dtype, int(value.size))

from train_target_reference import LightGCNReference  # noqa: E402

model = LightGCNReference()
print('graph var', model.graph.shape if hasattr(model.graph, 'shape') else type(model.graph))
print('state keys', list(model.state_dict())[:5], len(model.state_dict()))
users, items = model.computer()
print('computer', users.shape, items.shape)
scores = users[jt.array(np.arange(16))] @ items.transpose(1, 0)
print('scores', scores.shape, float(np.abs(scores.numpy()).max()))
