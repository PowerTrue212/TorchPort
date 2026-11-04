"""Freeze the paired DSH none/full SimCSE BERT-base inference trials (workflow 4090 r1).

Uses the existing seed layout (`seed/SimCSE`): upstream files at the repository root, the
fixed `evaluation_entry.py`, and an empty `run.py` stub the Agent implements. New
workspaces and controller directories only; never starts an Agent.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wf_prepare as wf  # noqa: E402

BENCH = Path('D:/科研/SimCSE-Benchmark-v1')
SEED = BENCH / 'seed/SimCSE'
PROMPT = 'prompt-simcse-inference.md'
STUB = ('"""Implement the SimCSE BERT-base model in native Jittor here."""\n\n\n'
        'def load_model():\n'
        '    """Return a BERT-base Jittor module with upstream parameter names."""\n'
        '    raise NotImplementedError("Implement BERT-base in native Jittor")\n')


def task_template(condition: str) -> dict:
    return {
        'schema_version': 1,
        'task_id': 'simcse-unsup-bert-base-stsb-forward-v1',
        'source': {
            'repo': str(BENCH / 'source/SimCSE'),
            'entry': 'torchport_inference_adapter_r1.py',
            'model': 'TorchPortSimCSE',
            'device': 'cuda',
            'python': '/root/autodl-tmp/envs/source/bin/python',
            'env': {'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                    'TP_SIMCSE_DEPS': '/root/autodl-tmp/SimCSE-Benchmark-v1/deps'},
        },
        'target': {
            'repo': '',
            'entry': 'evaluation_entry.py',
            'model': 'TorchPortSimCSE',
            'device': 'cuda',
            'python': '/root/autodl-tmp/envs/target/bin/python',
            'env': {
                'JITTOR_HOME': f'/root/autodl-tmp/SimCSE-Benchmark-v1/jittor-cache-wf-r1-{condition}',
                'nvcc_path': '/usr/local/cuda-12.8/bin/nvcc',
                'conv_opt': '1',
                'use_mkl': '0',
                'use_cutt': '0',
                'python_config': '/root/autodl-tmp/envs/target/bin/python3.9-config',
                'PATH': '/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:'
                        '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            },
        },
        'inputs': [
            {'shape': [8, 64], 'dtype': 'int64',
             'path': str(BENCH / 'inference-r1/inputs/input_ids.npy')},
            {'shape': [8, 64], 'dtype': 'int64',
             'path': str(BENCH / 'inference-r1/inputs/attention_mask.npy')},
            {'shape': [8, 64], 'dtype': 'int64',
             'path': str(BENCH / 'inference-r1/inputs/token_type_ids.npy')},
        ],
        'weights': str(BENCH / 'assets/models/princeton-nlp--unsup-simcse-bert-base-uncased'
                            '/weights_fp32.npz'),
        'seed': 20261026,
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


def prepare(condition: str, port: int) -> dict:
    # r2: the first attempt copied the seed's own .git, so git init found a dirty history.
    candidate = BENCH / 'workspaces-wf-r1-r2' / condition
    controller = BENCH / 'controller' / f'simcse-wf-{condition}-r1'
    if candidate.exists():
        raise FileExistsError(candidate)
    if controller.exists():
        raise FileExistsError(controller)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SEED, candidate, ignore=shutil.ignore_patterns('.git'))
    (candidate / 'run.py').write_text(STUB, encoding='utf-8')
    seed = wf.commit_seed(candidate, 'Frozen SimCSE source, fixed entry and empty Jittor run.py')
    task = task_template(condition)
    remote = wf.remote_block(
        str(BENCH), '/root/autodl-tmp/SimCSE-Benchmark-v1/harness-wf-r1',
        '/root/autodl-tmp/SimCSE-Benchmark-v1/home',
        [str(BENCH), str(BENCH / 'assets/models')])
    return wf.write_controller(
        BENCH, 'simcse', condition, port, task, candidate=candidate, remote=remote,
        forbidden_extra=[str(BENCH / 'inference-r1'), str(BENCH / 'reference')],
        prompt_tag=PROMPT, candidate_entry='run.py') | {'seed_commit': seed}


def main() -> None:
    print(json.dumps([prepare('none', 18947), prepare('full', 18948)],
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
