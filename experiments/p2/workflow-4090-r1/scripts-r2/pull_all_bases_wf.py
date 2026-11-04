"""Pull every remote harness base belonging to one controller directory.

Each TaskSpec path (task-*.json, plus trial/delivery/official-task-*.json) maps to
<harness>/<sha256(path)[:12]> on the remote. Calls pull_trial_evidence_json_wf.py once per
existing base. Usage:
  pull_all_bases_wf.py CONTROLLER_DIR REMOTE_HARNESS_ROOT LABEL
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

PULL = Path('D:/科研/TorchPort/experiments/p2/workflow-4090-r1/scripts/'
            'pull_trial_evidence_json_wf.py')
SSH_ENV = str(Path.home() / '.tp_ssh3.env')


def base_of(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]


def main() -> None:
    controller = Path(sys.argv[1]).resolve()
    harness = sys.argv[2].rstrip('/')
    label = sys.argv[3]
    tasks = sorted(controller.glob('task-*.json'))
    official = sorted((controller / 'trial' / 'delivery').glob('official-task-*.json'))
    for path in tasks + official:
        digest = base_of(path)
        suffix = 'agent' if path.parent == controller else 'official'
        name = f'{label}-{path.stem.replace("task-", "")}-{suffix}'
        target = controller / f'remote-evidence-{name}'
        if target.exists():
            print('skip existing', name)
            continue
        env = {'TP_SSH_ENV_FILE': SSH_ENV, 'PATH': ''}
        import os
        command = [sys.executable, str(PULL), str(controller), f'{harness}/{digest}', name]
        result = subprocess.run(command, text=True, capture_output=True,
                                env={**os.environ, **env})
        print(name, '->', (result.stdout or '').strip()[-300:] or result.stderr.strip()[-300:])


if __name__ == '__main__':
    main()
