"""Freeze paired DSH none/full FNO pilot trials without starting either Agent."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BENCH = Path('D:/科研/FNO-Benchmark-v1')
ARCHIVE = BENCH / 'source-snapshot-a6648ee.tar.gz'
SOURCE = BENCH / 'source'
INPUTS = BENCH / 'controller/train-inputs-4090-r2'
BASELINE = BENCH / 'controller/target-reference-4090-r1/target-repeat-baseline-r1.npz'
ORACLE_DIGEST = 'f3d97351178dd9ffe9b2e8f8c6995175f4b6820d8e8ab32d7e5e248f2739d607'
ARCHIVE_SHA = '092e8f5d3f6ae4ada1437a99baf39b3a1b058c6fa6db95746e2731d994566e36'
BASELINE_SHA = 'f57d320cb660cf5e0668166973ebce6ad641351d3562a9f469bdec47067f9cc8'
sys.path.insert(0, str(ROOT / 'src'))
from torchport.contracts import load_task  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def git(candidate: Path, *args: str, env=None) -> str:
    return subprocess.run(['git', *args], cwd=candidate, env=env, check=True,
                          text=True, capture_output=True).stdout.strip()


def create_candidate(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=False)
    upstream = path / 'upstream'
    upstream.mkdir()
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        for member in archive.getmembers():
            target = (upstream / member.name).resolve()
            if upstream.resolve() not in (target, *target.parents):
                raise ValueError('source archive escapes upstream directory')
            if not (member.isfile() or member.isdir()):
                raise ValueError('source archive contains unsupported member')
        archive.extractall(upstream, filter='data')
    (path / 'evaluation_entry.py').write_text(
        '"""Implement native Jittor FNO training here."""\n'
        'def load_train_model():\n'
        '    raise NotImplementedError("Port FNO to Jittor")\n', encoding='utf-8')
    git(path, 'init', '--initial-branch=main')
    git(path, 'add', '.')
    dated = {**os.environ, 'GIT_AUTHOR_DATE': '2026-11-03T00:00:00Z',
             'GIT_COMMITTER_DATE': '2026-11-03T00:00:00Z'}
    git(path, '-c', 'user.name=TorchPort Pilot', '-c', 'user.email=pilot@local.invalid',
        'commit', '-m', 'Frozen FNO source and empty Jittor entry', env=dated)
    if git(path, 'status', '--porcelain'):
        raise ValueError('candidate seed is not clean')
    return git(path, 'rev-parse', 'HEAD')


def prepare(condition: str, port: int) -> dict:
    candidate = BENCH / 'workspaces-train-five-r1' / condition
    controller = BENCH / 'controller' / f'fno-train-{condition}-five-r1'
    if candidate.exists() or controller.exists():
        raise FileExistsError(candidate if candidate.exists() else controller)
    seed = create_candidate(candidate)
    controller.mkdir(parents=True)
    task = json.loads((BENCH / 'controller/train-five-4090-r1/task.json').read_text('utf-8'))
    task['source']['repo'] = str(SOURCE)
    task['target']['repo'] = str(candidate)
    task['target']['env'].update({
        'JITTOR_HOME': f'/root/autodl-tmp/FNO-Benchmark-v1/jittor-cache-trial-r1-{condition}',
        'HOME': '/root/autodl-tmp/FNO-Benchmark-v1/home',
        'no_fuse': '1', 'DISABLE_MULTIPROCESSING': '1',
        'use_mkl': '0', 'use_cutt': '0',
    })
    task['inputs'][0]['path'] = str(INPUTS / 'x.npy')
    task['inputs'][1]['path'] = str(INPUTS / 'y.npy')
    task['weights'] = str(INPUTS / 'initial-weights.npz')
    task['protocol'].update({
        'train_envelope': 'train-repeat-envelope-v1-pilot',
        'train_noise_baseline': str(BASELINE),
        'train_noise_baseline_sha256': BASELINE_SHA,
    })
    task_path = controller / 'task.json'
    write_json(task_path, task)
    actual = load_task(task_path)['digest']
    if actual != ORACLE_DIGEST:
        raise ValueError(f'source oracle identity changed: {actual}')
    env_file = Path.home() / '.tp_ssh3.env'
    if not env_file.is_file():
        raise FileNotFoundError(env_file)
    trial_id = f'fno-train-{condition}-five-r1'
    write_json(controller / 'mcp-trial.json', {
        'schema_version': 1, 'status': 'pilot_only',
        'project_root': str(ROOT), 'candidate_root': str(candidate),
        'tasks': {'training': str(task_path)}, 'hidden_tasks': {'training': str(task_path)},
        'evidence_root': str(controller / 'evidence'), 'trial_dir': str(controller / 'trial'),
        'trial_id': trial_id, 'agent': 'DSH',
        'feedback': 'diagnostic' if condition == 'none' else 'localized',
        'max_oracle_queries': 10, 'max_wall_seconds': 10800,
        'candidate_entry': 'evaluation_entry.py',
        'remote': {
            'ssh_env_file': str(env_file),
            'root': '/root/autodl-tmp/FNO-Benchmark-v1/harness-train-five-r1',
            'source_python': '/root/autodl-tmp/envs/source/bin/python',
            'target_python': '/root/autodl-tmp/envs/target/bin/python',
            'python_config': '/root/autodl-tmp/envs/target/bin/python3.9-config',
            'system_libstdcxx': '',
            'nvcc_path': '/usr/local/cuda-12.8/bin/nvcc',
            'home': '/root/autodl-tmp/FNO-Benchmark-v1/home',
            'target_path': '/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:'
                           '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            'extra_roots': [str(BENCH)],
        },
    })
    token = secrets.token_urlsafe(32)
    (controller / 'mcp-token.txt').write_text(token + '\n', encoding='utf-8')
    (controller / 'dsh.patch.yml').write_text(
        '- id: sandbox-policy\n  config:\n    mode: workspace-write\n'
        '    workspaceRoot: "/work"\n- insert:\n'
        "    - id: mcp-migration-check\n      name: '@deepseek-ai/dsh-mcp-client'\n"
        '      config:\n        serverName: migration-check\n'
        '        transport: streamable-http\n'
        f'        url: "http://127.0.0.1:{port}/mcp"\n'
        f'        headers:\n          Authorization: "Bearer {token}"\n'
        '        toolCallTimeoutMs: 3900000\n        failOnStartupError: true\n',
        encoding='utf-8',
    )
    other = 'full' if condition == 'none' else 'none'
    forbidden = [str(ROOT), str(BENCH / 'controller'), str(SOURCE),
                 str(BENCH / 'workspaces-train-five-r1' / other),
                 str(env_file), str(Path.home() / '.codex')]
    write_json(controller / 'preflight.json', {
        'workspace': '/work', 'forbidden_paths': ['/mnt/c', '/mnt/d', '/root', '/host'],
        'forbidden_imports': ['torchport', 'experiments.harness', 'jittor_harness'],
        'advisory_urls': ['https://github.com/PowerTrue212/TorchPort'],
        'required_urls': ['https://pypi.org/simple/jittor/'],
        'required_tcp': [f'127.0.0.1:{port}'],
    })
    write_json(controller / 'audit-rules.json', {
        'forbidden_paths': forbidden + ['/mnt/c/', '/mnt/d/'],
        'forbidden_urls': ['powertrue212/jittor-harness', 'powertrue212/torchport',
                           'qinxi1477/torchport'],
        'allowed_paths': ['/usr/', '/home/node/', '/tmp/', '/proc/', '/etc/'],
        'max_steps': 200,
    })
    write_json(controller / 'container-trial.json', {'port': port, 'workspace': '/work'})
    return {'condition': condition, 'controller': str(controller), 'candidate': str(candidate),
            'seed_commit': seed, 'oracle_digest': actual,
            'feedback': 'diagnostic' if condition == 'none' else 'localized'}


def main() -> None:
    if sha(ARCHIVE) != ARCHIVE_SHA or sha(BASELINE) != BASELINE_SHA:
        raise ValueError('frozen source or baseline hash changed')
    print(json.dumps([prepare('none', 18931), prepare('full', 18932)], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
