"""Freeze a fresh RAFT full r5 workspace after the trace volume probe."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, 'D:/科研/TorchPort/experiments/p2/workflow-4090-r1/scripts-r2')
import wf_prepare_r2 as wf  # noqa: E402

RAFT = Path('D:/科研/RAFT-Benchmark-v1')
RAFT_TAR = RAFT / 'materials/raft-upstream-2888e15.tar.gz'
RAFT_TAR_SHA = 'be527f8676313236ceca2d39c93df8aedf34cd57a0e765126f7b833c305b7b55'
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


def main() -> None:
    if wf.sha(RAFT_TAR) != RAFT_TAR_SHA:
        raise ValueError('frozen RAFT archive changed')
    candidate = RAFT / 'workspaces-wf-r5' / 'full'
    controller = RAFT / 'controller' / 'raft-wf-full-r5'
    if candidate.exists() or controller.exists():
        raise FileExistsError('r5 candidate or controller already exists')
    candidate.mkdir(parents=True)
    wf.extract_upstream(RAFT_TAR, candidate / 'upstream')
    (candidate / 'evaluation_entry.py').write_text(RAFT_ENTRY, encoding='utf-8')
    seed = wf.commit_seed(candidate, 'Frozen RAFT source and empty Jittor entry')
    if seed != '2f8bcb53cfc476e43a0db83c18329320c585c266':
        raise ValueError(f'r5 seed differs from paired RAFT trials: {seed}')
    tasks = {}
    for name in ('inference', 'training'):
        task = wf.read_json(RAFT / 'controller' / 'raft-wf-full-r4' / f'task-{name}.json')
        task = json.loads(json.dumps(task))
        task['target']['repo'] = str(candidate)
        task['target']['env']['JITTOR_HOME'] = (
            '/root/autodl-tmp/RAFT-Benchmark-v1/jittor-cache-wf-r5-full'
        )
        tasks[name] = task
    prior = wf.read_json(RAFT / 'controller' / 'raft-wf-full-r4' / 'mcp-trial.json')
    remote = dict(prior['remote'])
    remote['root'] = '/root/autodl-tmp/RAFT-Benchmark-v1/harness-wf-r5'
    result = wf.write_controller(
        RAFT, 'raft', 'full', 18967, tasks, candidate=candidate, remote=remote,
        forbidden_extra=[str(RAFT / 'materials'), str(RAFT / 'source')],
        candidate_entry='evaluation_entry.py', max_oracle_queries=15,
        max_wall_seconds=14400, prompt_tag='prompt-raft.md',
        workspace_root='workspaces-wf-r5', controller_name='raft-wf-full-r5',
    )
    result['seed_commit'] = seed
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
