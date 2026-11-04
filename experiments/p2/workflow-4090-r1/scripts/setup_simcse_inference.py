"""Build the SimCSE BERT-base inference TaskSpec on the remote host (new inference-r1 tree).

Prerequisites (already uploaded):
  /root/autodl-tmp/SimCSE-Benchmark-v1/assets/unsup-pytorch_model.bin
  /root/autodl-tmp/SimCSE-Benchmark-v1/assets/unsup-config.json
  .../inference-r1/source/SimCSE/            (frozen source snapshot with torchport_adapter.py)
  .../inference-r1/SimCSE-Jittor/            (copy of reference/SimCSE-Jittor)
  .../inference-r1/inputs/{0,1,2}-*.npy      (8 fixed STS-B sentences, 64 tokens)
Creates: weights.npz (HF names), reference-r1 (copy of the Jittor reference plus run.py entry),
neg-bias-001 (hidden states x 1.001), task-r1.json. New files only.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

R = Path('/root/autodl-tmp/SimCSE-Benchmark-v1')
INF = R / 'inference-r1'
SRC = INF / 'source/SimCSE'
JT = INF / 'SimCSE-Jittor'
SOURCE_PY = '/root/autodl-tmp/envs/source/bin/python'
TARGET_PY = '/root/autodl-tmp/envs/target/bin/python'

RUN_PY = '''"""Controller-only Jittor BERT entry for the frozen unsupervised SimCSE checkpoint."""
from pathlib import Path

import jittor as jt
import numpy as np

from bert_jt import BertConfig, BertModel

WEIGHTS = Path('/root/autodl-tmp/SimCSE-Benchmark-v1/inference-r1/weights.npz')


class BertHidden(jt.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel(BertConfig(), add_pooling_layer=True)
        with np.load(WEIGHTS) as archive:
            state = {f'model.{name}': jt.array(np.asarray(archive[name])) for name in archive.files}
        self.model.load_state_dict(state, strict=False)

    def load_parameters(self, values):
        mapped = {name: jt.array(np.asarray(value)) for name, value in values.items()}
        self.load_state_dict(mapped, strict=False)

    def execute(self, input_ids, attention_mask, token_type_ids):
        out = self.model(input_ids, attention_mask, token_type_ids)
        return out['last_hidden_state']


def load_model():
    return BertHidden()
'''

ENTRY_PY = '''"""Fixed Jittor entry for the unsupervised SimCSE BERT-base inference pilot."""

from __future__ import annotations

import jittor as jt
from run import load_model


class TorchPortSimCSE(jt.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = load_model()

    def execute(self, input_ids, attention_mask, token_type_ids):
        chunks = []
        for start in range(0, input_ids.shape[0], 64):
            stop = start + 64
            hidden_states = self.model(input_ids[start:stop], attention_mask[start:stop],
                                       token_type_ids[start:stop])
            chunks.append(hidden_states[:, 0, :])
        return jt.concat(chunks, dim=0)
'''


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding='utf-8')


def main() -> None:
    checkpoint = R / 'assets/unsup-pytorch_model.bin'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    arrays = {name: value.detach().cpu().numpy() for name, value in state.items()}
    weights = INF / 'weights.npz'
    if weights.exists():
        raise FileExistsError(weights)
    np.savez(weights, **arrays)
    print('weights', len(arrays), 'names', list(arrays)[:3])

    ref = INF / 'reference-r1'
    ref.mkdir(parents=True, exist_ok=True)
    for name in ('bert_jt.py', 'model.py', 'dataset.py'):
        shutil.copy2(JT / name, ref / name)
    write(ref / 'run.py', RUN_PY)
    write(ref / 'evaluation_entry.py', ENTRY_PY)

    neg = INF / 'neg-bias-001'
    neg.mkdir(parents=True, exist_ok=True)
    for name in ('bert_jt.py', 'model.py', 'dataset.py', 'run.py', 'evaluation_entry.py'):
        shutil.copy2(ref / name, neg / name)
    text = (neg / 'run.py').read_text(encoding='utf-8')
    old = "        return out['last_hidden_state']"
    if text.count(old) != 1:
        raise ValueError('negative control edit point not unique')
    (neg / 'run.py').write_text(
        text.replace(old, "        return out['last_hidden_state'] * 1.001"), encoding='utf-8')

    task = {
        'schema_version': 1,
        'task_id': 'simcse-bert-base-inference-4090-pilot-r1',
        'source': {
            'repo': str(SRC),
            'entry': 'torchport_adapter.py',
            'model': 'TorchPortSimCSE',
            'device': 'cuda',
            'python': SOURCE_PY,
            'env': {'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                    'TP_SIMCSE_DEPS': str(R / 'deps')},
        },
        'target': {
            'repo': str(ref),
            'entry': 'evaluation_entry.py',
            'model': 'load_inference_model',
            'device': 'cuda',
            'python': TARGET_PY,
            'env': {
                'JITTOR_HOME': str(R / 'jittor-cache-inference-r1'),
                'nvcc_path': '/usr/local/cuda-12.8/bin/nvcc',
                'conv_opt': '1',
                'use_mkl': '0',
                'use_cutt': '0',
                'python_config': '/root/autodl-tmp/envs/target/bin/python3.9-config',
                'python_config_path': '/root/autodl-tmp/envs/target/bin/python3.9-config',
                'PATH': '/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:'
                        '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            },
        },
        'inputs': [
            {'shape': [8, 64], 'dtype': 'int64', 'path': str(INF / 'inputs/0-input_ids.npy')},
            {'shape': [8, 64], 'dtype': 'int64', 'path': str(INF / 'inputs/1-attention_mask.npy')},
            {'shape': [8, 64], 'dtype': 'int64', 'path': str(INF / 'inputs/2-token_type_ids.npy')},
        ],
        'weights': str(weights),
        'seed': 20261102,
        'protocol': {
            'profile': 'noise-envelope-v3',
            'repeats': 2,
            'timeout': 3600,
            'pass_k': 8.0,
            'floor_k': 8.0,
            'noise_floor': 0.0,
            'ignore_state': ['position_ids'],
            'variants': [{'name': 'fp64', 'dtype': 'float64'},
                         {'name': 'benchmark', 'cudnn_benchmark': True},
                         {'name': 'channels_last', 'channels_last': True}],
        },
    }
    write(INF / 'task-r1.json', json.dumps(task, indent=2) + '\n')
    print(json.dumps({
        'weights_sha256': sha(weights),
        'reference_run_sha256': sha(ref / 'run.py'),
        'negative_sha256': sha(neg / 'run.py'),
        'task': str(INF / 'task-r1.json'),
    }, indent=2))


if __name__ == '__main__':
    main()
