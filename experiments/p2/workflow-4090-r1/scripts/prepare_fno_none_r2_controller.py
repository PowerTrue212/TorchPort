"""Recreate the FNO none controller after a `--skip-agent` dry run planted canaries.

The dry run wrote controller/fno-wf-none-r1/canaries.json, and plant() refuses to reuse a
controller that already has canaries. The empty Agent workspace is unchanged, so this only
builds a fresh controller directory (r2) bound to the same workspace and task content.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wf_prepare as wf  # noqa: E402

BENCH = Path('D:/科研/FNO-Benchmark-v1')
CANDIDATE = BENCH / 'workspaces-wf-r1/none'


def main() -> None:
    task = json.loads((BENCH / 'controller/fno-wf-none-r1/task.json').read_text(encoding='utf-8'))
    remote = wf.remote_block(
        str(BENCH), '/root/autodl-tmp/FNO-Benchmark-v1/harness-wf-r1',
        '/root/autodl-tmp/FNO-Benchmark-v1/home', [str(BENCH)])
    result = wf.write_controller(
        BENCH, 'fno', 'none', 18943, task, candidate=CANDIDATE, remote=remote,
        forbidden_extra=[str(BENCH / 'inference-r1')], prompt_tag='prompt-fno-inference.md',
        candidate_entry='evaluation_entry.py', controller_name='fno-wf-none-r3')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
