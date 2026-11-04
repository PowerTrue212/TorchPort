"""Probe the FNO source state_dict structure (which entries are not tensors)."""
import os
import sys

R = '/root/autodl-tmp/FNO-Benchmark-v1'
os.environ['TP_FNO_DEPS'] = f'{R}/deps-requests:{R}/deps-pinned'
sys.path[:0] = [f'{R}/source', *os.environ['TP_FNO_DEPS'].split(os.pathsep)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchport_inference_adapter_r1 import load_inference_model  # noqa: E402

model = load_inference_model()
state = model.state_dict()
print('entries', len(state))
for name, value in state.items():
    if not hasattr(value, 'detach'):
        print('NON-TENSOR', name, type(value), list(value)[:6] if hasattr(value, '__iter__') else '')
print('torch', torch.__version__)
raw = np.load(f'{R}/inference-r1/weights.npz')
print('weights', len(raw.files), 'first', raw.files[0])
