"""Collect the workflow-4090-r1 trial numbers and print the phase-5 table rows."""
from __future__ import annotations

import json
from pathlib import Path

TRIALS = [
    ('RAFT 推理+训练', 'none', None, '阶段 1 门槛未过（1.2 留一 1/12 误拒、1.3 neg-wd 未被拒）'),
    ('RAFT 推理+训练', 'full', None, '阶段 1 门槛未过（同上）'),
    ('FNO 推理', 'none', Path('D:/科研/FNO-Benchmark-v1/controller/fno-wf-none-r3'), ''),
    ('FNO 推理', 'full', Path('D:/科研/FNO-Benchmark-v1/controller/fno-wf-full-r1'), ''),
    ('LightGCN 推理', 'none', None, '阶段 2 门槛未过（ORACLE_UNSTABLE：source sparse.mm 非确定）'),
    ('LightGCN 推理', 'full', None, '阶段 2 门槛未过（同上）'),
    ('SimCSE 推理', 'none', Path('D:/科研/SimCSE-Benchmark-v1/controller/simcse-wf-none-r1'), ''),
    ('SimCSE 推理', 'full', Path('D:/科研/SimCSE-Benchmark-v1/controller/simcse-wf-full-r1'), ''),
]


def read(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def official_code(controller: Path) -> str:
    report = read(controller / 'trial' / 'official-verify.json')
    if not report:
        return ''
    codes = []
    for name, value in report.items():
        codes.append(str(value.get('failure_code') or ('PASS' if value.get('passed') else '?')))
    return ','.join(codes)


def main() -> None:
    print('| 仓库 | 组 | passed | failure_code | logical query | physical 执行次数 | agent 墙钟(s) | 模型步数 | 备注 |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for repo, condition, controller, skip in TRIALS:
        if controller is None:
            print(f'| {repo} | {condition} | - | SKIP | - | - | - | - | {skip} |')
            continue
        result = read(controller / 'trial' / 'result.json')
        audit = read(controller / 'audit.json')
        runtime = read(controller / 'container-runtime.json')
        if result is None:
            state = '未完成'
            print(f'| {repo} | {condition} | {state} | - | - | - | - | - | {skip} |')
            continue
        logical = (result.get('logical_query_cost') or {}).get('total')
        physical = (result.get('physical_evaluation_cost') or {}).get('total', {}).get('executions')
        wall = result.get('agent_wall_seconds')
        steps = (audit or {}).get('model_steps')
        passed = result.get('passed')
        code = official_code(controller) or str(result.get('stop_reason'))
        note = skip or result.get('stop_reason', '')
        if runtime and runtime.get('cleanup_error'):
            note = (note + ' cleanup_error').strip()
        print(f'| {repo} | {condition} | {passed} | {code} | {logical} | {physical} | '
              f'{round(wall or 0, 1)} | {steps} | {note} |')


if __name__ == '__main__':
    main()
