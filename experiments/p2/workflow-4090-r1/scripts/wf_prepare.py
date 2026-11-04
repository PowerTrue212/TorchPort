"""Shared helpers for the 4090 r1 paired DSH inference trials (none/full).

Only new directories are created; an existing workspace or controller directory aborts.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path('D:/科研/TorchPort')
ENV_FILE = Path.home() / '.tp_ssh3.env'
DATE = '2026-11-04T00:00:00Z'

sys.path.insert(0, str(ROOT / 'src'))
from torchport.contracts import load_task  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def git(candidate: Path, *args: str, env=None) -> str:
    return subprocess.run(['git', *args], cwd=candidate, env=env, check=True,
                          text=True, capture_output=True).stdout.strip()


def commit_seed(candidate: Path, message: str) -> str:
    git(candidate, 'init', '--initial-branch=main')
    git(candidate, 'add', '.')
    dated = {**os.environ, 'GIT_AUTHOR_DATE': DATE, 'GIT_COMMITTER_DATE': DATE}
    git(candidate, '-c', 'user.name=TorchPort Pilot', '-c', 'user.email=pilot@local.invalid',
        'commit', '-m', message, env=dated)
    if git(candidate, 'status', '--porcelain'):
        raise ValueError(f'candidate seed is not clean: {candidate}')
    return git(candidate, 'rev-parse', 'HEAD')


def extract_upstream(archive: Path, upstream: Path) -> None:
    upstream.mkdir()
    with tarfile.open(archive, 'r:gz') as package:
        for member in package.getmembers():
            target = (upstream / member.name).resolve()
            if upstream.resolve() not in (target, *target.parents):
                raise ValueError('source archive escapes upstream directory')
            if not (member.isfile() or member.isdir()):
                raise ValueError('source archive contains unsupported member')
        package.extractall(upstream, filter='data')


def pair_dirs(bench: Path, repo: str, condition: str) -> tuple[Path, Path]:
    candidate = bench / 'workspaces-wf-r1' / condition
    controller = bench / 'controller' / f'{repo}-wf-{condition}-r1'
    if candidate.exists():
        raise FileExistsError(candidate)
    if controller.exists():
        raise FileExistsError(controller)
    return candidate, controller


def write_controller(
    bench: Path, repo: str, condition: str, port: int, task: dict, *,
    candidate: Path, remote: dict, forbidden_extra: list[str], prompt_tag: str,
    candidate_entry: str, controller_name: str | None = None,
) -> dict:
    name = controller_name or f'{repo}-wf-{condition}-r1'
    controller = bench / 'controller' / name
    if controller.exists():
        raise FileExistsError(controller)
    controller.mkdir(parents=True)
    task_path = controller / 'task.json'
    task['target']['repo'] = str(candidate)
    write_json(task_path, task)
    load_task(task_path)  # validate
    evidence_root = controller / 'evidence'
    trial_dir = controller / 'trial'
    trial_id = name
    write_json(controller / 'mcp-trial.json', {
        'schema_version': 1, 'status': 'pilot_only',
        'project_root': str(ROOT), 'candidate_root': str(candidate),
        'tasks': {'inference': str(task_path)}, 'hidden_tasks': {'inference': str(task_path)},
        'evidence_root': str(evidence_root), 'trial_dir': str(trial_dir),
        'trial_id': trial_id, 'agent': 'DSH',
        'feedback': 'diagnostic' if condition == 'none' else 'localized',
        'max_oracle_queries': 10, 'max_wall_seconds': 7200,
        'candidate_entry': candidate_entry,
        'remote': remote,
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
    forbidden = [str(ROOT), str(bench / 'controller'), str(bench / 'source'),
                 str(bench / 'workspaces-wf-r1' / other), str(ENV_FILE),
                 str(Path.home() / '.codex'), *forbidden_extra]
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
            'feedback': 'diagnostic' if condition == 'none' else 'localized',
            'task_digest': load_task(task_path)['digest'], 'prompt': prompt_tag}


def remote_block(repo_dir: str, harness: str, home: str, extra_roots: list[str]) -> dict:
    return {
        'ssh_env_file': str(ENV_FILE),
        'root': harness,
        'source_python': '/root/autodl-tmp/envs/source/bin/python',
        'target_python': '/root/autodl-tmp/envs/target/bin/python',
        'python_config': '/root/autodl-tmp/envs/target/bin/python3.9-config',
        'system_libstdcxx': '',
        'nvcc_path': '/usr/local/cuda-12.8/bin/nvcc',
        'home': home,
        'target_path': '/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:'
                       '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        'extra_roots': extra_roots,
    }
