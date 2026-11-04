"""Run one hidden verify for an inference TaskSpec (noise-envelope-v3).

Binds only target.repo when --repo is given; writes a new task file (mode 'x') next to the
report so nothing frozen is overwritten.

Usage: run_inference_verify.py TASK_JSON EVIDENCE_ROOT TRIAL [--repo DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torchport.contracts import load_task
from torchport.verification import verify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('task')
    parser.add_argument('evidence')
    parser.add_argument('trial')
    parser.add_argument('--repo', default=None)
    args = parser.parse_args()
    if not args.trial.replace('-', '').isalnum():
        raise ValueError('trial')
    root = Path(args.evidence)
    root.mkdir(parents=True, exist_ok=True)
    if args.repo:
        raw = json.loads(Path(args.task).read_text())
        raw['target']['repo'] = str(Path(args.repo).resolve())
        task_path = root / f'task-{args.trial}.json'
        with task_path.open('x') as stream:
            json.dump(raw, stream, indent=2)
    else:
        task_path = Path(args.task)
    report = verify(load_task(task_path), root, trial=args.trial)
    e1 = report.get('e1') or {}
    print(json.dumps({
        'trial': args.trial,
        'task_digest': report.get('task_digest'),
        'passed': report.get('passed'),
        'e1_status': report.get('e1_status'),
        'failure_code': report.get('failure_code'),
        'summary': report.get('summary'),
        'metrics': e1.get('metrics'),
        'first_divergence': e1.get('first_divergence'),
        'repeats': report.get('repeats'),
        'physical': report.get('physical_cost'),
    }, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
