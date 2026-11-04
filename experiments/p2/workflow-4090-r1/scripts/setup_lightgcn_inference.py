"""Build the LightGCN LastFM inference TaskSpec on the remote host (new inference-r1 tree).

Prerequisite: the frozen source snapshot has been extracted to
/root/autodl-tmp/LightGCN-Benchmark-v1/inference-r1/source/LightGCN-PyTorch-pilot-r2 and
the checkpoint to .../inference-r1/weights.npz.
Creates only new files: fixed 16 user/item ids, a NEW copy of the Jittor reference with
load_inference_model(), a source inference adapter, a negative control (output x 1.001),
and the noise-envelope-v3 task.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

R = Path('/root/autodl-tmp/LightGCN-Benchmark-v1')
INF = R / 'inference-r1'
REF = R / 'controller/target-reference-r2'
SRC = INF / 'source/LightGCN-PyTorch-pilot-r2'
SOURCE_PY = '/root/autodl-tmp/envs/source/bin/python'
TARGET_PY = '/root/autodl-tmp/envs/target/bin/python'


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
    for sub in ('inputs', 'reference-r1', 'neg-bias-001', 'scripts'):
        (INF / sub).mkdir(parents=True, exist_ok=False)
    users = np.arange(16, dtype=np.int64)
    items = np.arange(16, dtype=np.int64)
    np.save(INF / 'inputs' / '0-users.npy', users)
    np.save(INF / 'inputs' / '1-items.npy', items)

    for name in ('train_target_reference.py', 'graph-r1.npz'):
        shutil.copy2(REF / name, INF / 'reference-r1' / name)
    write(INF / 'reference-r1' / 'evaluation_entry.py',
          '"""Controller-only Jittor LightGCN inference entry (copy of the frozen reference).\n\n'
          'Same model as train_target_reference.load_train_model, in eval mode.\n'
          '"""\n'
          'from train_target_reference import LightGCNReference\n\n\n'
          'def load_inference_model():\n'
          '    return LightGCNReference()\n')

    for name in ('train_target_reference.py', 'graph-r1.npz', 'evaluation_entry.py'):
        shutil.copy2(INF / 'reference-r1' / name, INF / 'neg-bias-001' / name)
    neg = INF / 'neg-bias-001' / 'train_target_reference.py'
    text = neg.read_text(encoding='utf-8')
    old = '        return (user_table[users] * item_table[items]).sum(dim=1)'
    if text.count(old) != 1:
        raise ValueError('negative control edit point not unique')
    neg.write_text(
        text.replace(old, '        return (user_table[users] * item_table[items]).sum(dim=1) * 1.001'),
        encoding='utf-8')

    write(SRC / 'torchport_inference_adapter_r1.py',
          '"""Controller source entry for the frozen LastFM LightGCN inference checkpoint."""\n'
          'from torchport_train_adapter import load_train_model  # noqa: E402\n\n\n'
          'def load_inference_model():\n'
          '    return load_train_model()\n')

    task = {
        'schema_version': 1,
        'task_id': 'lightgcn-lastfm-inference-4090-pilot-r1',
        'source': {
            'repo': str(SRC),
            'entry': 'torchport_inference_adapter_r1.py',
            'model': 'load_inference_model',
            'device': 'cuda',
            'python': SOURCE_PY,
            'env': {'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                    'TP_LIGHTGCN_DEPS': str(R / 'deps'),
                    'HOME': str(R / 'home')},
        },
        'target': {
            'repo': str(INF / 'reference-r1'),
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
                'HOME': str(R / 'home'),
                'PATH': '/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:'
                        '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            },
        },
        'inputs': [
            {'shape': [16], 'dtype': 'int64', 'path': str(INF / 'inputs' / '0-users.npy')},
            {'shape': [16], 'dtype': 'int64', 'path': str(INF / 'inputs' / '1-items.npy')},
        ],
        'weights': str(INF / 'weights.npz'),
        'seed': 20261102,
        'protocol': {
            'profile': 'noise-envelope-v3',
            'repeats': 2,
            'timeout': 1800,
            'pass_k': 8.0,
            'floor_k': 8.0,
            'noise_floor': 0.0,
            'variants': [{'name': 'fp64', 'dtype': 'float64'},
                         {'name': 'benchmark', 'cudnn_benchmark': True},
                         {'name': 'channels_last', 'channels_last': True}],
        },
    }
    write(INF / 'task-r1.json', json.dumps(task, indent=2) + '\n')
    print(json.dumps({
        'weights_sha256': sha(INF / 'weights.npz'),
        'reference_entry_sha256': sha(INF / 'reference-r1' / 'evaluation_entry.py'),
        'negative_sha256': sha(INF / 'neg-bias-001' / 'train_target_reference.py'),
        'source_adapter_sha256': sha(SRC / 'torchport_inference_adapter_r1.py'),
        'task': str(INF / 'task-r1.json'),
    }, indent=2))


if __name__ == '__main__':
    main()
