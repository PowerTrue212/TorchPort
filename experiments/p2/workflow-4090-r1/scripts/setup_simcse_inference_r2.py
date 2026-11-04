"""SimCSE setup r2: deps-aware source entry + task-r2 (task-r1 and its runs are kept)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

R = Path('/root/autodl-tmp/SimCSE-Benchmark-v1')
INF = R / 'inference-r1'
ADAPTER = INF / 'source/SimCSE/torchport_inference_adapter_r1.py'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not ADAPTER.is_file():
        raise FileNotFoundError(ADAPTER)
    task = json.loads((INF / 'task-r1.json').read_text())
    task['source']['entry'] = 'torchport_inference_adapter_r1.py'
    out = INF / 'task-r2.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(task, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'adapter_sha256': sha(ADAPTER), 'task': str(out)}, indent=2))


if __name__ == '__main__':
    main()
