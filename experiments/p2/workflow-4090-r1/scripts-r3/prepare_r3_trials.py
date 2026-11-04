"""Freeze workflow r3: RAFT none/full (inference + training) and FNO inference none.

r3 differs from r2 only in what the r2 audit found wrong:
- RAFT training task adds protocol.marginal_rerun = {upper 3.0, runs 5}; the shared-parameter
  canonical-name fix is in the committed TorchPort code (edb09fb), bundled by the harness.
- FNO none starts again from the frozen seed commit (r5 resumed r4's half-finished workspace).
- Fresh workspaces, controllers, harness root and per-condition Jittor caches.
Prompts are unchanged. New directories only; an existing one aborts the script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts-r2'))
import wf_prepare_r2 as wf  # noqa: E402

RAFT = Path('D:/科研/RAFT-Benchmark-v1')
RAFT_ARCHIVE = RAFT / 'materials/raft-upstream-2888e15.tar.gz'
RAFT_ARCHIVE_SHA = 'be527f8676313236ceca2d39c93df8aedf34cd57a0e765126f7b833c305b7b55'
RAFT_BASELINE = RAFT / 'controller/cal-r2/target-repeat-baseline.npz'
RAFT_BASELINE_SHA = '2c8048de9b10d812258cc64116f357ac5102364d847169977a97ad7f05241e38'
RAFT_TRAIN_DIGEST = '5586ca6424603914791b15252cfd26782f93064a35494f954f07c2610d3999fd'
RAFT_SEED = '2f8bcb53cfc476e43a0db83c18329320c585c266'

FNO = Path('D:/科研/FNO-Benchmark-v1')
FNO_ARCHIVE = FNO / 'source-snapshot-a6648ee.tar.gz'
FNO_ARCHIVE_SHA = '092e8f5d3f6ae4ada1437a99baf39b3a1b058c6fa6db95746e2731d994566e36'
FNO_TASK = FNO / 'controller/fno-wf-full-r1/task.json'
FNO_SEED = '3bc3200662f36a36683ec7bb593f0c541519ea44'

RAFT_ENTRY = '''"""Implement the native Jittor RAFT reference here (inference and three-step training)."""


def load_inference_model():
    raise NotImplementedError("Port RAFT inference to Jittor")


def load_train_model():
    raise NotImplementedError("Port RAFT training to Jittor")


def build_optimizer(model):
    raise NotImplementedError("Port the RAFT optimizer to Jittor")


def train_step(model, optimizer, image1, image2, flow_gt, valid):
    raise NotImplementedError("Port the RAFT train step to Jittor")
'''
FNO_STUB = ('"""Implement native Jittor FNO inference here."""\n'
            'def load_inference_model():\n'
            '    raise NotImplementedError("Port FNO inference to Jittor")\n')


def seed_workspace(candidate: Path, archive: Path, entry: str, message: str, expected: str,
                   date: str) -> str:
    candidate.mkdir(parents=True, exist_ok=False)
    wf.extract_upstream(archive, candidate / 'upstream')
    (candidate / 'evaluation_entry.py').write_text(entry, encoding='utf-8')
    # the seed commit hash includes the commit date; reuse each repository's earlier date so
    # the r3 seed is byte-identical to the r1/r2 seed
    wf.DATE = date
    seed = wf.commit_seed(candidate, message)
    if seed != expected:
        raise ValueError(f'seed commit differs from earlier rounds: {seed} != {expected}')
    return seed


def raft(condition: str, port: int) -> dict:
    base = wf.read_json(RAFT / f'controller/raft-wf-{condition}-r1/task-training.json')
    infer = wf.read_json(RAFT / f'controller/raft-wf-{condition}-r1/task-inference.json')
    candidate = RAFT / 'workspaces-wf-r3' / condition
    seed = seed_workspace(candidate, RAFT_ARCHIVE, RAFT_ENTRY,
                          'Frozen RAFT source and empty Jittor entry', RAFT_SEED,
                          '2026-11-05T00:00:00Z')
    cache = f'/root/autodl-tmp/RAFT-Benchmark-v1/jittor-cache-wf-r3-{condition}'
    for task in (base, infer):
        task['target']['env']['JITTOR_HOME'] = cache
    base['protocol'].update({
        'train_noise_baseline': str(RAFT_BASELINE),
        'train_noise_baseline_sha256': RAFT_BASELINE_SHA,
        'marginal_rerun': {'upper': 3.0, 'runs': 5},
    })
    record = wf.write_controller(
        RAFT, 'raft', condition, port, {'inference': infer, 'training': base},
        candidate=candidate,
        remote=wf.remote_block('/root/autodl-tmp/RAFT-Benchmark-v1/harness-wf-r3',
                               '/root/autodl-tmp/RAFT-Benchmark-v1/home', [str(RAFT)]),
        forbidden_extra=[str(RAFT / 'materials'), str(RAFT / 'source'),
                         str(RAFT / 'workspaces-wf-r1')],
        candidate_entry='evaluation_entry.py', max_oracle_queries=15,
        max_wall_seconds=14400, workspace_root='workspaces-wf-r3',
        prompt_tag='prompt-raft.md', controller_name=f'raft-wf-{condition}-r3')
    if record['digests']['training'] != RAFT_TRAIN_DIGEST:
        raise ValueError(f"RAFT training oracle identity changed: {record['digests']['training']}")
    return record | {'seed_commit': seed}


def fno_none(port: int) -> dict:
    task = wf.read_json(FNO_TASK)
    candidate = FNO / 'workspaces-wf-r3' / 'none'
    seed = seed_workspace(candidate, FNO_ARCHIVE, FNO_STUB,
                          'Frozen FNO source and empty Jittor inference entry', FNO_SEED,
                          '2026-11-04T00:00:00Z')
    task['target']['env']['JITTOR_HOME'] = '/root/autodl-tmp/FNO-Benchmark-v1/jittor-cache-wf-r3-none'
    record = wf.write_controller(
        FNO, 'fno', 'none', port, {'inference': task},
        candidate=candidate,
        remote=wf.remote_block('/root/autodl-tmp/FNO-Benchmark-v1/harness-wf-r3',
                               '/root/autodl-tmp/FNO-Benchmark-v1/home', [str(FNO)]),
        forbidden_extra=[str(FNO / 'inference-r1'), str(FNO / 'workspaces-wf-r1')],
        candidate_entry='evaluation_entry.py', max_oracle_queries=10,
        max_wall_seconds=7200, workspace_root='workspaces-wf-r3',
        prompt_tag='prompt-fno-inference.md', controller_name='fno-wf-none-r6')
    if record['digests']['inference'] != wf.load_task(FNO_TASK)['digest']:
        raise ValueError('FNO inference oracle identity differs from the full-r1 trial')
    return record | {'seed_commit': seed}


def main() -> None:
    for path, expected in ((RAFT_ARCHIVE, RAFT_ARCHIVE_SHA), (RAFT_BASELINE, RAFT_BASELINE_SHA),
                           (FNO_ARCHIVE, FNO_ARCHIVE_SHA)):
        if wf.sha(path) != expected:
            raise ValueError(f'frozen file changed: {path}')
    records = [raft('none', 18961), raft('full', 18962), fno_none(18963)]
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
