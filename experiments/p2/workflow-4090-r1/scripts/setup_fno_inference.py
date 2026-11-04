"""Build the FNO Darcy inference TaskSpec on the remote host (new inference-r1 tree).

Creates only new files/directories: fixed 4-sample input, a copy of the frozen checkpoint,
a NEW copy of the Jittor reference with load_inference_model(), a source inference adapter,
and the noise-envelope-v3 task. Nothing already there is rewritten.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

R = Path('/root/autodl-tmp/FNO-Benchmark-v1')
INF = R / 'inference-r1'
REF_SRC = R / 'controller-spectral-r1'
METRICS = R / 'task-metrics-r2'
SOURCE_PY = '/root/autodl-tmp/envs/source/bin/python'
TARGET_PY = '/root/autodl-tmp/envs/target/bin/python'
DEPS = f'{R}/deps-requests:{R}/deps-pinned'


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
    test = np.load(METRICS / 'test-inputs.npy')
    x = np.ascontiguousarray(test[:4].astype(np.float32))
    np.save(INF / 'inputs' / '0-x.npy', x)
    shutil.copy2(METRICS / 'weights.npz', INF / 'weights.npz')

    for name in ('fno_reference.py', 'spectral.py'):
        shutil.copy2(REF_SRC / name, INF / 'reference-r1' / name)
    write(INF / 'reference-r1' / 'evaluation_entry.py',
          '"""Controller-only Jittor FNO inference entry (copy of the frozen reference).\n\n'
          'Same model as fno_reference.load_train_model, in eval mode; never shipped to an Agent.\n'
          '"""\n'
          'from fno_reference import FNOReference\n\n\n'
          'def load_inference_model():\n'
          '    return FNOReference()\n')

    # negative control: identical copy, output scaled by 1.001
    for name in ('fno_reference.py', 'spectral.py', 'evaluation_entry.py'):
        shutil.copy2(INF / 'reference-r1' / name, INF / 'neg-bias-001' / name)
    neg = INF / 'neg-bias-001' / 'fno_reference.py'
    text = neg.read_text(encoding='utf-8')
    old = "        return self.mlp(x, 'projection')"
    if text.count(old) != 1:
        raise ValueError('negative control edit point not unique')
    neg.write_text(text.replace(old, "        return self.mlp(x, 'projection') * 1.001"), encoding='utf-8')

    adapter = R / 'source' / 'torchport_inference_adapter_r1.py'
    write(adapter,
          '"""Controller source entry for the frozen Darcy FNO inference checkpoint."""\n'
          'import os\n'
          'import sys\n\n'
          'if os.environ.get("TP_FNO_DEPS"):\n'
          '    sys.path[:0] = os.environ["TP_FNO_DEPS"].split(os.pathsep)\n\n'
          'from torchport_train_adapter import SourceFNO  # noqa: E402\n\n\n'
          'def load_inference_model() -> SourceFNO:\n'
          '    return SourceFNO(\n'
          '        n_modes=(4, 4),\n'
          '        in_channels=1,\n'
          '        out_channels=1,\n'
          '        hidden_channels=8,\n'
          '        n_layers=2,\n'
          '    )\n')

    task = {
        'schema_version': 1,
        'task_id': 'fno-darcy16-inference-4090-pilot-r1',
        'source': {
            'repo': str(R / 'source'),
            'entry': 'torchport_inference_adapter_r1.py',
            'model': 'load_inference_model',
            'device': 'cuda',
            'python': SOURCE_PY,
            'env': {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'TP_FNO_DEPS': DEPS},
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
                'no_fuse': '1',
                'DISABLE_MULTIPROCESSING': '1',
                'python_config': '/root/autodl-tmp/envs/target/bin/python3.9-config',
                'python_config_path': '/root/autodl-tmp/envs/target/bin/python3.9-config',
                'PATH': '/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:'
                        '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            },
        },
        'inputs': [{'shape': [4, 1, 16, 16], 'dtype': 'float32',
                    'path': str(INF / 'inputs' / '0-x.npy')}],
        'weights': str(INF / 'weights.npz'),
        'seed': 20261030,
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
        'input': {'shape': list(x.shape), 'sha256': sha(INF / 'inputs' / '0-x.npy')},
        'weights_sha256': sha(INF / 'weights.npz'),
        'reference_sha256': sha(INF / 'reference-r1' / 'evaluation_entry.py'),
        'negative_sha256': sha(INF / 'neg-bias-001' / 'fno_reference.py'),
        'task': str(INF / 'task-r1.json'),
    }, indent=2))


if __name__ == '__main__':
    main()
