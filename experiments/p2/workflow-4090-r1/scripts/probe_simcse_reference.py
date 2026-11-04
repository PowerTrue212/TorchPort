"""Probe the SimCSE controller reference without TorchPort.

Loads the frozen checkpoint through the entry, prints state-dict names and one forward
shape, then loads oracle-style parameters from the source state naming convention.
"""
import sys
from pathlib import Path

import jittor as jt
import numpy as np

REF = Path('/root/autodl-tmp/SimCSE-Benchmark-v1/inference-r1/reference-r3')
INF = Path('/root/autodl-tmp/SimCSE-Benchmark-v1/inference-r1')
sys.path[:0] = [str(REF)]

from evaluation_entry import load_inference_model  # noqa: E402

jt.flags.use_cuda = 1
model = load_inference_model()
model.eval()
names = list(model.state_dict())
print('state entries', len(names))
print('first', names[:3])
print('pooler', [n for n in names if 'pooler' in n])
inputs = [jt.array(np.load(INF / 'inputs' / name)) for name in
          ('0-input_ids.npy', '1-attention_mask.npy', '2-token_type_ids.npy')]
out = model(*inputs)
print('output', out.shape, out.dtype, float(jt.abs(out).max().numpy()))
print('first row head', out.numpy()[0, :4])
