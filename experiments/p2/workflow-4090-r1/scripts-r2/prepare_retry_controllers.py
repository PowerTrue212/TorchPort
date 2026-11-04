"""Create retry controllers after the docker/proxy infrastructure failure at 11:03.

The three failed trials stopped in preflight (pypi unreachable through the dead local
proxy), so no Agent turn ever started. New controller directories bind the same tasks and
the same workspaces: fno-wf-none-r4, lightgcn-wf-none-r2, lightgcn-wf-full-r2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wf_prepare_r2 as wf  # noqa: E402

FNO = Path('D:/科研/FNO-Benchmark-v1')
LGCN = Path('D:/科研/LightGCN-Benchmark-v1')


def main() -> None:
    results = []
    fno_task = wf.read_json(FNO / 'controller/fno-wf-none-r3/task.json')
    results.append(wf.write_controller(
        FNO, 'fno', 'none', 18943, {'inference': fno_task},
        candidate=FNO / 'workspaces-wf-r1/none',
        remote=wf.remote_block('/root/autodl-tmp/FNO-Benchmark-v1/harness-wf-r1',
                               '/root/autodl-tmp/FNO-Benchmark-v1/home', [str(FNO)]),
        forbidden_extra=[str(FNO / 'inference-r1')],
        candidate_entry='evaluation_entry.py', max_oracle_queries=10,
        max_wall_seconds=7200, prompt_tag='prompt-fno-inference.md',
        controller_name='fno-wf-none-r5'))

    for condition, port in (('none', 18953), ('full', 18954)):
        task = wf.read_json(LGCN / f'controller/lightgcn-wf-{condition}-r1/task-inference.json')
        results.append(wf.write_controller(
            LGCN, 'lightgcn', condition, port, {'inference': task},
            candidate=LGCN / 'workspaces-wf-r1' / condition,
            remote=wf.remote_block('/root/autodl-tmp/LightGCN-Benchmark-v1/harness-wf-r1',
                                   '/root/autodl-tmp/LightGCN-Benchmark-v1/home', [str(LGCN)]),
            forbidden_extra=[str(LGCN / 'materials'), str(LGCN / 'inference-r1')],
            candidate_entry='evaluation_entry.py', max_oracle_queries=10,
            max_wall_seconds=7200, prompt_tag='prompt-lightgcn-inference.md',
            controller_name=f'lightgcn-wf-{condition}-r2'))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
