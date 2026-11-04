"""Pull one trial's remote harness directory with a SHA256 manifest and archive hash.

Remote base = /root/autodl-tmp/<repo>-Benchmark-v1/harness-wf-r1/<sha256(task path)[:12]>.
Usage: pull_trial_evidence_wf.py CONTROLLER_DIR REMOTE_HARNESS_ROOT NAME
"""
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, 'D:/科研/TorchPort/scripts/remote')
from ssh_exec import connect  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def run(client, command: str) -> str:
    _, out, err = client.exec_command(command, timeout=3600)
    text = out.read().decode('utf-8', 'replace')
    if out.channel.recv_exit_status():
        raise RuntimeError(err.read().decode('utf-8', 'replace'))
    return text


def main() -> None:
    controller = Path(sys.argv[1]).resolve()
    harness = sys.argv[2].rstrip('/')
    name = sys.argv[3]
    task_path = controller / 'task.json'
    digest = hashlib.sha256(str(task_path).encode()).hexdigest()[:12]
    base = f'{harness}/{digest}'
    local = controller / f'remote-evidence-{name}'
    if local.exists():
        raise FileExistsError(local)
    archive = f'/root/autodl-tmp/{name}.tar.gz'
    client = connect()
    try:
        exists = run(client, f'test -d {base} && echo yes || echo no').strip()
        if exists != 'yes':
            raise FileNotFoundError(base)
        run(client, f'test ! -e {archive}')
        run(client, f'cd {base} && find . -type f -print0 | xargs -0 sha256sum > ../{name}.sha256 '
                    f'&& tar -czf {archive} . 2>/dev/null')
        remote_sha = run(client, f'sha256sum {archive}').split()[0]
        local.mkdir(parents=True)
        local_tar = local / f'{name}.tar.gz'
        sftp = client.open_sftp()
        sftp.get(archive, str(local_tar))
        sftp.close()
        manifest_text = run(client, f'cat {base}/../{name}.sha256')
    finally:
        client.close()
    local_sha = sha256(local_tar)
    if local_sha != remote_sha:
        raise RuntimeError(f'archive hash mismatch: {local_sha} != {remote_sha}')
    (local / f'{name}.sha256').write_text(manifest_text, encoding='utf-8')
    with tarfile.open(local_tar) as tar:
        tar.extractall(local / 'raw', filter='data')
    bad = []
    for line in manifest_text.splitlines():
        digest_hex, _, rel = line.partition('  ')
        target = local / 'raw' / rel.lstrip('./')
        if not target.is_file() or sha256(target) != digest_hex:
            bad.append(rel)
    report = {'archive': str(local_tar), 'archive_sha256': local_sha,
              'remote_base': base, 'files': len(manifest_text.splitlines()), 'bad': bad}
    (local / 'verify.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({**report, 'bad': len(bad)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
