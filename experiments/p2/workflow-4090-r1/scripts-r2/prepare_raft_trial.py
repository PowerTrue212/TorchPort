"""Freeze the paired DSH none/full RAFT inference+training trials (workflow 4090 r2).

One trial carries two hidden tasks (inference + training). New directories only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wf_prepare_r2 as wf  # noqa: E402

BENCH = Path('D:/科研/RAFT-Benchmark-v1')
ARCHIVE = BENCH / 'materials/raft-upstream-2888e15.tar.gz'
ARCHIVE_SHA = 'be527f8676313236ceca2d39c93df8aedf34cd57a0e765126f7b833c305b7b55'
BASELINE = BENCH / 'controller/cal-r2/target-repeat-baseline.npz'
BASELINE_SHA = '2c8048de9b10d812258cc64116f357ac5102364d847169977a97ad7f05241e38'
TRAIN_TEMPLATE = BENCH / 'controller/train-three-r1/task.json'
INFER_TEMPLATE = BENCH / 'controller/inference-r1/task.json'
ORACLE_DIGEST = '5586ca6424603914791b15252cfd26782f93064a35494f954f07c2610d3999fd'
PROMPT = 'prompt-raft.md'

ENTRY = '''"""Implement the native Jittor RAFT reference here (inference and three-step training)."""


def load_inference_model():
    raise NotImplementedError("Port RAFT inference to Jittor")


def load_train_model():
    raise NotImplementedError("Port RAFT training to Jittor")


def build_optimizer(model):
    raise NotImplementedError("Port the RAFT optimizer to Jittor")


def train_step(model, optimizer, image1, image2, flow_gt, valid):
    raise NotImplementedError("Port the RAFT train step to Jittor")
'''


def target_block(condition: str) -> dict:
    return {
        'repo': '',
        'entry': 'evaluation_entry.py',
        'model': '',
        'device': 'cuda',
        'python': '/root/autodl-tmp/envs/target/bin/python',
        'env': {
            'JITTOR_HOME': f'/root/autodl-tmp/RAFT-Benchmark-v1/jittor-cache-wf-r1-{condition}',
            'nvcc_path': '/usr/local/cuda-12.8/bin/nvcc',
            'conv_opt': '1',
            'use_mkl': '0',
            'use_cutt': '0',
            'python_config': '/root/autodl-tmp/envs/target/bin/python3.9-config',
            'python_config_path': '/root/autodl-tmp/envs/target/bin/python3.9-config',
            'PATH': '/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:'
                    '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        },
    }


def main() -> None:
    if wf.sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('frozen RAFT source archive changed')
    if wf.sha(BASELINE) != BASELINE_SHA:
        raise ValueError('frozen RAFT repeat baseline changed')
    train_base = wf.read_json(TRAIN_TEMPLATE)
    infer_base = wf.read_json(INFER_TEMPLATE)
    results = []
    for condition, port in (('none', 18951), ('full', 18952)):
        candidate = BENCH / 'workspaces-wf-r1' / condition
        controller = BENCH / 'controller' / f'raft-wf-{condition}-r1'
        if candidate.exists():
            raise FileExistsError(candidate)
        if controller.exists():
            raise FileExistsError(controller)
        candidate.mkdir(parents=True)
        wf.extract_upstream(ARCHIVE, candidate / 'upstream')
        (candidate / 'evaluation_entry.py').write_text(ENTRY, encoding='utf-8')
        seed = wf.commit_seed(candidate, 'Frozen RAFT source and empty Jittor entry')

        training = json.loads(json.dumps(train_base))
        training['target'] = {**target_block(condition), 'model': 'load_train_model'}
        training['protocol'].update({
            'train_envelope': 'train-repeat-envelope-v1-pilot',
            'train_noise_baseline': str(BASELINE),
            'train_noise_baseline_sha256': BASELINE_SHA,
        })
        inference = json.loads(json.dumps(infer_base))
        inference['target'] = {**target_block(condition), 'model': 'load_inference_model'}
        inference['protocol'] = {
            'profile': 'noise-envelope-v3', 'repeats': 2, 'timeout': 1800,
            'pass_k': 8.0, 'floor_k': 8.0, 'noise_floor': 0.0,
            'variants': [{'name': 'fp64', 'dtype': 'float64'},
                         {'name': 'benchmark', 'cudnn_benchmark': True},
                         {'name': 'channels_last', 'channels_last': True}],
        }
        remote = wf.remote_block(
            '/root/autodl-tmp/RAFT-Benchmark-v1/harness-wf-r1',
            '/root/autodl-tmp/RAFT-Benchmark-v1/home', [str(BENCH)])
        record = wf.write_controller(
            BENCH, 'raft', condition, port,
            {'inference': inference, 'training': training},
            candidate=candidate, remote=remote,
            forbidden_extra=[str(BENCH / 'materials'), str(BENCH / 'source')],
            candidate_entry='evaluation_entry.py',
            max_oracle_queries=15, max_wall_seconds=14400, prompt_tag=PROMPT)
        if record['digests']['training'] != ORACLE_DIGEST:
            raise ValueError(f"training oracle identity changed: {record['digests']['training']}")
        record['seed_commit'] = seed
        results.append(record)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
