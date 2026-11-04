"""Freeze the paired DSH none/full FNO Darcy inference trials (workflow 4090 r1).

Creates new workspaces and controller directories only; never starts an Agent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wf_prepare as wf  # noqa: E402

BENCH = Path('D:/科研/FNO-Benchmark-v1')
ARCHIVE = BENCH / 'source-snapshot-a6648ee.tar.gz'
ARCHIVE_SHA = '092e8f5d3f6ae4ada1437a99baf39b3a1b058c6fa6db95746e2731d994566e36'
PROMPT = 'prompt-fno-inference.md'

STUB = ('"""Implement native Jittor FNO inference here."""\n'
        'def load_inference_model():\n'
        '    raise NotImplementedError("Port FNO inference to Jittor")\n')


def task_template(condition: str) -> dict:
    return {
        'schema_version': 1,
        'task_id': 'fno-darcy16-inference-4090-pilot-r1',
        'source': {
            'repo': str(BENCH / 'source'),
            'entry': 'torchport_inference_adapter_r2.py',
            'model': 'load_inference_model',
            'device': 'cuda',
            'python': '/root/autodl-tmp/envs/source/bin/python',
            'env': {'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                    'TP_FNO_DEPS': '/root/autodl-tmp/FNO-Benchmark-v1/deps-requests:'
                                   '/root/autodl-tmp/FNO-Benchmark-v1/deps-pinned'},
        },
        'target': {
            'repo': '',
            'entry': 'evaluation_entry.py',
            'model': 'load_inference_model',
            'device': 'cuda',
            'python': '/root/autodl-tmp/envs/target/bin/python',
            'env': {
                'JITTOR_HOME': f'/root/autodl-tmp/FNO-Benchmark-v1/jittor-cache-wf-r1-{condition}',
                'nvcc_path': '/usr/local/cuda-12.8/bin/nvcc',
                'conv_opt': '1',
                'use_mkl': '0',
                'use_cutt': '0',
                'no_fuse': '1',
                'DISABLE_MULTIPROCESSING': '1',
                'python_config': '/root/autodl-tmp/envs/target/bin/python3.9-config',
                'PATH': '/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:'
                        '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            },
        },
        'inputs': [{'shape': [4, 1, 16, 16], 'dtype': 'float32',
                    'path': str(BENCH / 'inference-r1/inputs/0-x.npy')}],
        'weights': str(BENCH / 'inference-r1/weights.npz'),
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


def prepare(condition: str, port: int) -> dict:
    candidate, _ = wf.pair_dirs(BENCH, 'fno', condition)
    candidate.mkdir(parents=True)
    wf.extract_upstream(ARCHIVE, candidate / 'upstream')
    (candidate / 'evaluation_entry.py').write_text(STUB, encoding='utf-8')
    seed = wf.commit_seed(candidate, 'Frozen FNO source and empty Jittor inference entry')
    task = task_template(condition)
    remote = wf.remote_block(
        str(BENCH), '/root/autodl-tmp/FNO-Benchmark-v1/harness-wf-r1',
        '/root/autodl-tmp/FNO-Benchmark-v1/home', [str(BENCH)])
    return wf.write_controller(
        BENCH, 'fno', condition, port, task, candidate=candidate, remote=remote,
        forbidden_extra=[str(BENCH / 'inference-r1')], prompt_tag=PROMPT,
        candidate_entry='evaluation_entry.py') | {'seed_commit': seed}


def main() -> None:
    if wf.sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('frozen FNO source archive changed')
    print(json.dumps([prepare('none', 18943), prepare('full', 18944)],
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
