"""Freeze the paired DSH none/full LightGCN inference trials (workflow 4090 r2) - r2 fix.

r1 run failed because it looked for the dense adapter under the TorchPort tree; Claude's
scripts live in D:\\科研\\Workflow-4090-r1\\scripts-r2. The source copy was already created
(without the adapter), so this run tolerates that and only adds the missing adapter file.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wf_prepare_r2 as wf  # noqa: E402

BENCH = Path('D:/科研/LightGCN-Benchmark-v1')
SNAPSHOT = BENCH / 'source-snapshot-947ca2b3.tar.gz'
SNAPSHOT_SHA = '7b21fc57dfbbfb1a74e16e729b0ef45da878af54fdf2d18cb10fca262275e948'
SRC_BASE = BENCH / 'source/LightGCN-PyTorch-pilot-r2'
SRC_DENSE = BENCH / 'source/LightGCN-PyTorch-dense-r1'
ADAPTER = Path('D:/科研/Workflow-4090-r1/scripts-r2/torchport_inference_adapter_r3.py')
ASSETS = BENCH / 'inference-wf-r2'
TEMPLATE = BENCH / 'controller/r2-templates/task-r3.json'
PROMPT = 'prompt-lightgcn-inference.md'
SSH = 'D:/科研/TorchPort/scripts/remote/ssh_exec.py'
REMOTE_DIR = '/root/autodl-tmp/LightGCN-Benchmark-v1/inference-r1'

ENTRY = '''"""Implement native Jittor LightGCN inference here."""


def load_inference_model():
    raise NotImplementedError("Port LightGCN inference to Jittor")
'''


def ssh(*args: str) -> str:
    env = {**os.environ, 'TP_SSH_ENV_FILE': str(wf.ENV_FILE)}
    result = subprocess.run([sys.executable, SSH, *args], text=True, capture_output=True, env=env)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def prepare_assets() -> dict[str, str]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    remote_hashes = ssh('run', f'sha256sum {REMOTE_DIR}/inputs/0-users.npy {REMOTE_DIR}/weights.npz')
    expected = {}
    for line in remote_hashes.splitlines():
        digest, _, path = line.partition('  ')
        if path.strip() and '/' in path:
            expected[Path(path.strip()).name] = digest.strip()
    for local_name, remote in (('0-users.npy', f'{REMOTE_DIR}/inputs/0-users.npy'),
                               ('weights.npz', f'{REMOTE_DIR}/weights.npz')):
        local = ASSETS / local_name
        if local.exists():
            raise FileExistsError(local)
        ssh('get', remote, str(local))
        if wf.sha(local) != expected[local_name]:
            raise ValueError(f'remote/local hash mismatch: {local_name}')
    return expected


def prepare_source() -> None:
    if not SRC_DENSE.exists():
        shutil.copytree(SRC_BASE, SRC_DENSE, ignore=shutil.ignore_patterns('__pycache__'))
    target = SRC_DENSE / ADAPTER.name
    if not target.exists():
        shutil.copy2(ADAPTER, target)
    if wf.sha(target) != wf.sha(ADAPTER):
        raise ValueError('dense adapter copy differs')


def main() -> None:
    if wf.sha(SNAPSHOT) != SNAPSHOT_SHA:
        raise ValueError('frozen LightGCN source snapshot changed')
    prepare_source()
    hashes = prepare_assets()
    template = wf.read_json(TEMPLATE)
    results = []
    for condition, port in (('none', 18953), ('full', 18954)):
        candidate = BENCH / 'workspaces-wf-r1' / condition
        controller = BENCH / 'controller' / f'lightgcn-wf-{condition}-r1'
        if candidate.exists():
            raise FileExistsError(candidate)
        if controller.exists():
            raise FileExistsError(controller)
        candidate.mkdir(parents=True)
        wf.extract_upstream(SNAPSHOT, candidate / 'upstream')
        (candidate / 'evaluation_entry.py').write_text(ENTRY, encoding='utf-8')
        seed = wf.commit_seed(candidate, 'Frozen LightGCN source and empty Jittor entry')

        task = json.loads(json.dumps(template))
        task['source']['repo'] = str(SRC_DENSE)
        task['source']['entry'] = ADAPTER.name
        task['inputs'] = [{'shape': [16], 'dtype': 'int64',
                           'path': str(ASSETS / '0-users.npy')}]
        task['weights'] = str(ASSETS / 'weights.npz')
        task['target']['model'] = 'load_inference_model'
        task['target']['env']['JITTOR_HOME'] = (
            f'/root/autodl-tmp/LightGCN-Benchmark-v1/jittor-cache-wf-r2-{condition}')
        remote = wf.remote_block(
            '/root/autodl-tmp/LightGCN-Benchmark-v1/harness-wf-r1',
            '/root/autodl-tmp/LightGCN-Benchmark-v1/home', [str(BENCH)])
        record = wf.write_controller(
            BENCH, 'lightgcn', condition, port, {'inference': task},
            candidate=candidate, remote=remote,
            forbidden_extra=[str(BENCH / 'materials'), str(BENCH / 'inference-r1')],
            candidate_entry='evaluation_entry.py',
            max_oracle_queries=10, max_wall_seconds=7200, prompt_tag=PROMPT)
        record['seed_commit'] = seed
        record['assets_sha256'] = hashes
        results.append(record)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
