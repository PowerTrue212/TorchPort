"""LightGCN setup r4: single-input (user ids -> ratings over all items) inference task.

r1 used two inputs (users, items) and per-pair scores, which contradicts the prompt's
contract (`load_inference_model()` called with a user-id vector returns [len(users), 4489]).
r4 keeps r1's evidence and adds reference-r2, neg-bias-r2 and task-r2.json.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

R = Path('/root/autodl-tmp/LightGCN-Benchmark-v1')
INF = R / 'inference-r1'
REF1 = INF / 'reference-r1'
SRC = INF / 'source/LightGCN-PyTorch-pilot-r2'

ENTRY = '''"""Controller-only Jittor LightGCN inference entry (ratings over all items)."""

from __future__ import annotations

import jittor as jt
from train_target_reference import LightGCNReference


class Ratings(jt.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = LightGCNReference()

    def state_dict(self):
        return self.model.state_dict()

    def load_parameters(self, values):
        self.model.load_parameters(values)

    def execute(self, users):
        user_table, item_table = self.model.computer()
        scores = user_table[users] @ item_table.transpose(1, 0)
        return jt.sigmoid(scores)


def load_inference_model():
    return Ratings()
'''


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ref = INF / 'reference-r2'
    if ref.exists():
        raise FileExistsError(ref)
    shutil.copytree(REF1, ref)
    (ref / 'evaluation_entry.py').write_text(ENTRY, encoding='utf-8')

    neg = INF / 'neg-bias-r2'
    if neg.exists():
        raise FileExistsError(neg)
    shutil.copytree(ref, neg)
    text = (neg / 'evaluation_entry.py').read_text(encoding='utf-8')
    old = '        return jt.sigmoid(scores)'
    if text.count(old) != 1:
        raise ValueError('negative control edit point not unique')
    (neg / 'evaluation_entry.py').write_text(
        text.replace(old, '        return jt.sigmoid(scores) * 1.001'), encoding='utf-8')

    adapter = SRC / 'torchport_inference_adapter_r2.py'
    if not adapter.is_file():
        raise FileNotFoundError(adapter)

    task = json.loads((INF / 'task-r1.json').read_text())
    task['source']['entry'] = 'torchport_inference_adapter_r2.py'
    task['source']['model'] = 'load_inference_model'
    task['target']['repo'] = str(ref)
    task['inputs'] = [{'shape': [16], 'dtype': 'int64',
                       'path': str(INF / 'inputs/0-users.npy')}]
    out = INF / 'task-r2.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(task, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'entry_sha256': sha(ref / 'evaluation_entry.py'),
                      'negative_sha256': sha(neg / 'evaluation_entry.py'),
                      'adapter_sha256': sha(adapter),
                      'task': str(out)}, indent=2))


if __name__ == '__main__':
    main()
