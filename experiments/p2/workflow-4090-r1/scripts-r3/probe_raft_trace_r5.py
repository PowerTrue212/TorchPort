"""Measure focused RAFT reference trace size on 4090, isolated from trial evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path

from torchport.contracts import load_task
from torchport.diagnosis import trace
from torchport.runtime import write_json

BASE = Path('/root/autodl-tmp/RAFT-Benchmark-v1')
BUNDLE = BASE / 'oracle-bundle-inference-r1-r6'
ROOT = BASE / 'trace-probe-r5' / 'evidence'
WINDOW = os.environ.get('TRACE_PROBE_WINDOW', 'last')
TRIAL = f'reference-{WINDOW}-window-r5'


def main() -> None:
    task = load_task(BUNDLE / 'task-reference-v3.json')
    prior = json.loads((BUNDLE / 'evidence-v3/verify/v3-reference-r1.json').read_text())
    e1 = prior['e1']
    eligible = [t for t in e1['tensors'] if isinstance(t.get('position'), int)
                and not str(t.get('name', '')).startswith('output')]
    ordered = sorted(eligible, key=lambda t: t['position'])
    last = ordered[-1] if WINDOW == 'last' else ordered[int(WINDOW)]
    # Reference passed; this synthetic focus marker is only a volume probe.
    write_json(ROOT / 'verify' / f'{TRIAL}.json', {
        'kind': 'torchport.verify.volume-probe', 'trial_id': TRIAL,
        'e1': {**e1, 'first_divergence': {'name': last['name'], 'position': last['position']}},
    })
    result = trace(task, ROOT, trial=TRIAL)
    sizes = {}
    for side in ('source_run', 'target_run'):
        run = result.get(side) or {}
        path = Path(run['dir']) if run.get('dir') else None
        sizes[side] = {
            'ok': run.get('ok'),
            'bytes': sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) if path and path.exists() else 0,
            'contract': run.get('contract'),
        }
    write_json(BASE / 'trace-probe-r5' / f'summary-{WINDOW}.json', {
        'trial_id': TRIAL, 'focus_tensor': last, 'focus': result.get('focus'),
        'failure_code': result.get('failure_code'), 'passed': result.get('passed'),
        'sizes': sizes, 'report': result.get('artifacts', {}).get('report'),
    })
    print(json.dumps({'failure_code': result.get('failure_code'), 'sizes': sizes,
                      'focus_tensor': last['name']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
