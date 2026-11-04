"""Warm one JITTOR_HOME with the frozen RAFT training reference (no candidate code)."""
import sys
from pathlib import Path

import jittor as jt
import numpy as np

R = Path('/root/autodl-tmp/RAFT-Benchmark-v1')
BUNDLE = R / 'oracle-bundle-train-three-r1-r6'
sys.path[:0] = [str(R / 'controller-reference-r1')]

import evaluation_entry as ref  # noqa: E402

jt.flags.use_cuda = 1
model = ref.load_train_model()
optimizer = ref.build_optimizer(model)
names = ('0-train-image1.npy', '1-train-image2.npy', '2-train-flow.npy', '3-train-valid.npy')
values = [jt.array(np.load(BUNDLE / 'inputs' / name)[0]) for name in names]
loss = ref.train_step(model, optimizer, *values)
print('loss', float(np.asarray(loss.numpy()).reshape(-1)[0]))
print('params', sum(int(np.prod(p.shape)) for _, p in model.named_parameters()))
