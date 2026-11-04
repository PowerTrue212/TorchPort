"""SimCSE setup r3: controller entry that keeps the source's parameter names.

r1's entry imported the agent-facing `load_model` and exposed no `load_inference_model`;
r3 wraps run.py's model as `self.model` but delegates state_dict/load_parameters so the
portable names stay `model.<bert name>` exactly like the source state.
Creates reference-r2, neg-bias-r2 (hidden states x 1.001) and task-r3.json.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

R = Path('/root/autodl-tmp/SimCSE-Benchmark-v1')
INF = R / 'inference-r1'
REF1 = INF / 'reference-r1'

ENTRY = '''"""Controller-only Jittor SimCSE inference entry (first-token hidden states)."""

from __future__ import annotations

import jittor as jt
from run import load_model


class FirstToken(jt.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = load_model()

    def state_dict(self):
        return self.model.state_dict()

    def load_parameters(self, values):
        self.model.load_parameters(values)

    def execute(self, input_ids, attention_mask, token_type_ids):
        chunks = []
        for start in range(0, input_ids.shape[0], 64):
            stop = start + 64
            hidden_states = self.model(input_ids[start:stop], attention_mask[start:stop],
                                       token_type_ids[start:stop])
            chunks.append(hidden_states[:, 0, :])
        return jt.concat(chunks, dim=0)


def load_inference_model():
    return FirstToken()
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
    text = (neg / 'run.py').read_text(encoding='utf-8')
    old = "        return out['last_hidden_state']"
    if text.count(old) != 1:
        raise ValueError('negative control edit point not unique')
    (neg / 'run.py').write_text(
        text.replace(old, "        return out['last_hidden_state'] * 1.001"), encoding='utf-8')

    task = json.loads((INF / 'task-r2.json').read_text())
    task['target']['repo'] = str(ref)
    out = INF / 'task-r3.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(task, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'entry_sha256': sha(ref / 'evaluation_entry.py'),
                      'negative_sha256': sha(neg / 'run.py'),
                      'task': str(out)}, indent=2))


if __name__ == '__main__':
    main()
