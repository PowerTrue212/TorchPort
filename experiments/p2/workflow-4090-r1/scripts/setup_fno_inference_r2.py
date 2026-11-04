"""Second FNO inference setup: source adapter that freezes tensors only, plus task-r2.

The remote copy of torchport_train_adapter.py is an older revision without the
`_metadata` pop, so save_state sees a dict entry and fails. This adapter subclasses
SourceFNO and drops every non-tensor entry from state_dict; tensors are unchanged.
New files only (task-r2.json, adapter r2); task-r1 and its evidence are kept.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

R = Path('/root/autodl-tmp/FNO-Benchmark-v1')
INF = R / 'inference-r1'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    adapter = R / 'source' / 'torchport_inference_adapter_r2.py'
    if adapter.exists():
        raise FileExistsError(adapter)
    adapter.write_text(
        '"""Controller source entry for the frozen Darcy FNO inference checkpoint (r2).\n\n'
        'Only non-tensor state entries (upstream `_metadata`) are dropped; every parameter and\n'
        'buffer is kept unchanged.\n'
        '"""\n'
        'import os\n'
        'import sys\n\n'
        'if os.environ.get("TP_FNO_DEPS"):\n'
        '    sys.path[:0] = os.environ["TP_FNO_DEPS"].split(os.pathsep)\n\n'
        'from torchport_train_adapter import SourceFNO  # noqa: E402\n\n\n'
        'class InferenceFNO(SourceFNO):\n'
        '    def state_dict(self, *args, **kwargs):\n'
        '        state = super().state_dict(*args, **kwargs)\n'
        '        return {name: value for name, value in state.items() if hasattr(value, "detach")}\n\n\n'
        'def load_inference_model() -> InferenceFNO:\n'
        '    return InferenceFNO(\n'
        '        n_modes=(4, 4),\n'
        '        in_channels=1,\n'
        '        out_channels=1,\n'
        '        hidden_channels=8,\n'
        '        n_layers=2,\n'
        '    )\n',
        encoding='utf-8')

    task = json.loads((INF / 'task-r1.json').read_text())
    task['source']['entry'] = 'torchport_inference_adapter_r2.py'
    out = INF / 'task-r2.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(task, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'adapter': str(adapter), 'adapter_sha256': sha(adapter),
                      'task': str(out)}, indent=2))


if __name__ == '__main__':
    main()
