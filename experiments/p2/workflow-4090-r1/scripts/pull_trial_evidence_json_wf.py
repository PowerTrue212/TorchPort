"""Pull the JSON/log part of one remote harness base plus a full SHA256 manifest.

The SimCSE bases hold 3-14 GB of layers.npz evidence; pulling all of it is impractical.
This pulls every non-.npz file and the remote manifest of *all* files, so the npz can be
re-fetched later with an exact hash contract. Usage:
  pull_trial_evidence_json_wf.py CONTROLLER_DIR REMOTE_BASE NAME
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
    base = sys.argv[2].rstrip('/')
    name = sys.argv[3]
    local = controller / f'remote-evidence-{name}'
    if local.exists():
        raise FileExistsError(local)
    archive = f'/root/autodl-tmp/{name}.tar.gz'
    manifest_remote = f'{base}/../{name}.sha256'
    client = connect()
    try:
        run(client, f'test -d {base}')
        run(client, f'cd {base} && find . -type f -print0 | xargs -0 sha256sum > {manifest_remote}')
        run(client, f'cd {base} && find . -type f ! -name "*.npz" -print0 | '
                    f'tar -czf {archive} --null -T -')
        remote_sha = run(client, f'sha256sum {archive}').split()[0]
        manifest_text = run(client, f'cat {manifest_remote}')
        local.mkdir(parents=True)
        local_tar = local / f'{name}-json.tar.gz'
        sftp = client.open_sftp()
        sftp.get(archive, str(local_tar))
        sftp.close()
    finally:
        client.close()
    local_sha = sha256(local_tar)
    if local_sha != remote_sha:
        raise RuntimeError(f'archive hash mismatch: {local_sha} != {remote_sha}')
    (local / f'{name}.sha256').write_text(manifest_text, encoding='utf-8')
    with tarfile.open(local_tar) as tar:
        tar.extractall(local / 'raw', filter='data')
    entries = [line.partition('  ') for line in manifest_text.splitlines()]
    pulled = {rel.lstrip('./'): digest for digest, _, rel in entries
              if (local / 'raw' / rel.lstrip('./')).is_file()}
    bad = [rel for rel, digest in pulled.items()
           if sha256(local / 'raw' / rel) != digest]
    npz = sum(1 for _, _, rel in entries if rel.endswith('.npz'))
    report = {'archive': str(local_tar), 'archive_sha256': local_sha, 'remote_base': base,
              'manifest_files': len(entries), 'pulled_files': len(pulled),
              'npz_not_pulled': npz, 'bad': bad}
    (local / 'verify.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({**report, 'bad': len(bad)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
