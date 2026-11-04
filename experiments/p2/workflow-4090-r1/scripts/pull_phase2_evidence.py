"""Pull the phase-2 inference verify reports and task files to the workflow evidence dir.

Only small JSON evidence is pulled (no npz, no Jittor caches).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, 'D:/科研/TorchPort/scripts/remote')
from ssh_exec import connect  # noqa: E402

TARGETS = {
    'fno': ['/root/autodl-tmp/FNO-Benchmark-v1/inference-r1/task-r1.json',
            '/root/autodl-tmp/FNO-Benchmark-v1/inference-r1/task-r2.json',
            '/root/autodl-tmp/FNO-Benchmark-v1/inference-r1/evidence/verify'],
    'lightgcn': ['/root/autodl-tmp/LightGCN-Benchmark-v1/inference-r1/task-r1.json',
                 '/root/autodl-tmp/LightGCN-Benchmark-v1/inference-r1/task-r2.json',
                 '/root/autodl-tmp/LightGCN-Benchmark-v1/inference-r1/evidence/verify'],
    'simcse': ['/root/autodl-tmp/SimCSE-Benchmark-v1/inference-r1/task-r4.json',
               '/root/autodl-tmp/SimCSE-Benchmark-v1/inference-r1/evidence/verify'],
}
LOCAL = Path('D:/科研/Workflow-4090-r1/phase2-evidence')


def main() -> None:
    LOCAL.mkdir(parents=True, exist_ok=True)
    client = connect()
    sftp = client.open_sftp()
    pulled = []
    try:
        for repo, paths in TARGETS.items():
            out = LOCAL / repo
            out.mkdir(exist_ok=True)
            for remote in paths:
                try:
                    with sftp.open(remote, 'rb') as handle:
                        stat = sftp.stat(remote)
                except IOError:
                    print('missing', remote)
                    continue
                if stat.st_mode & 0o040000:  # directory
                    for entry in sftp.listdir_attr(remote):
                        src = f'{remote}/{entry.filename}'
                        dest = out / f'{Path(remote).name}-{entry.filename}'
                        sftp.get(src, str(dest))
                        pulled.append(str(dest))
                else:
                    dest = out / Path(remote).name
                    sftp.get(remote, str(dest))
                    pulled.append(str(dest))
    finally:
        sftp.close()
        client.close()
    print(json.dumps({'pulled': pulled}, indent=1))


if __name__ == '__main__':
    main()
