"""SimCSE setup r4: run.py without the unsupported strict kwarg.

Jittor 1.3.11 Module.load_state_dict(state_dict) takes no `strict`. r4 loads the frozen
checkpoint and the oracle state by filtering to the module's own portable names.
Creates reference-r3, neg-bias-r3 (hidden states x 1.001) and task-r4.json.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

R = Path('/root/autodl-tmp/SimCSE-Benchmark-v1')
INF = R / 'inference-r1'
REF2 = INF / 'reference-r2'

RUN_PY = '''"""Controller-only Jittor BERT entry for the frozen unsupervised SimCSE checkpoint."""
from pathlib import Path

import jittor as jt
import numpy as np

from bert_jt import BertConfig, BertModel

WEIGHTS = Path('/root/autodl-tmp/SimCSE-Benchmark-v1/inference-r1/weights.npz')
PREFIX = 'model.'


class BertHidden(jt.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel(BertConfig(), add_pooling_layer=True)
        with np.load(WEIGHTS) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        known = set(self.model.state_dict())
        self.model.load_state_dict({name: jt.array(value) for name, value in arrays.items()
                                    if name in known})

    def load_parameters(self, values):
        known = set(self.model.state_dict())
        mapped = {}
        for name, value in values.items():
            key = name[len(PREFIX):] if name.startswith(PREFIX) else name
            if key in known:
                mapped[key] = jt.array(np.asarray(value))
        self.model.load_state_dict(mapped)

    def execute(self, input_ids, attention_mask, token_type_ids):
        out = self.model(input_ids, attention_mask, token_type_ids)
        return out['last_hidden_state']


def load_model():
    return BertHidden()
'''


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ref = INF / 'reference-r3'
    if ref.exists():
        raise FileExistsError(ref)
    shutil.copytree(REF2, ref)
    (ref / 'run.py').write_text(RUN_PY, encoding='utf-8')

    neg = INF / 'neg-bias-r3'
    if neg.exists():
        raise FileExistsError(neg)
    shutil.copytree(ref, neg)
    text = (neg / 'run.py').read_text(encoding='utf-8')
    old = "        return out['last_hidden_state']"
    if text.count(old) != 1:
        raise ValueError('negative control edit point not unique')
    (neg / 'run.py').write_text(
        text.replace(old, "        return out['last_hidden_state'] * 1.001"), encoding='utf-8')

    task = json.loads((INF / 'task-r3.json').read_text())
    task['target']['repo'] = str(ref)
    out = INF / 'task-r4.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(task, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'run_sha256': sha(ref / 'run.py'),
                      'negative_sha256': sha(neg / 'run.py'),
                      'task': str(out)}, indent=2))


if __name__ == '__main__':
    main()
