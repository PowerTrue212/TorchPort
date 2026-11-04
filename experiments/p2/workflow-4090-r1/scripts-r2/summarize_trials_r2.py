"""Collect the workflow-4090-r1 r2 trial numbers and print the phase-E table rows."""
from __future__ import annotations

import json
from pathlib import Path

TRIALS = [
    ('RAFT 推理+训练', 'none', Path('D:/科研/RAFT-Benchmark-v1/controller/raft-wf-none-r1'), ''),
    ('RAFT 推理+训练', 'full', Path('D:/科研/RAFT-Benchmark-v1/controller/raft-wf-full-r1'), ''),
    ('FNO 推理', 'none', Path('D:/科研/FNO-Benchmark-v1/controller/fno-wf-none-r3'), ''),
    ('LightGCN 推理', 'none', Path('D:/科研/LightGCN-Benchmark-v1/controller/lightgcn-wf-none-r1'), ''),
    ('LightGCN 推理', 'full', Path('D:/科研/LightGCN-Benchmark-v1/controller/lightgcn-wf-full-r1'), ''),
]


def read(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def official_codes(controller: Path) -> tuple[str, str, str]:
    report = read(controller / 'trial' / 'official-verify.json')
    if not report:
        return '', '', ''
    codes, worst, failed = [], [], 0
    for name, value in report.items():
        codes.append(f'{name}:{value.get("failure_code") or "PASS"}')
        e1 = (value.get('e1') or {}).get('metrics') or {}
        worst.append(f'{name} max_ratio={e1.get("max_ratio")}')
        failed += int(e1.get('failed_tensors') or 0)
    return ';'.join(codes), ';'.join(worst), str(failed)


def main() -> None:
    print('| 仓库 | 组 | passed | failure_code | logical query | physical 执行次数 | agent 墙钟(s) | 模型步数 | 超限张量/max ratio |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for repo, condition, controller, skip in TRIALS:
        result = read(controller / 'trial' / 'result.json')
        if result is None:
            print(f'| {repo} | {condition} | 未完成 | - | - | - | - | - | {skip or "无 result.json"} |')
            continue
        audit = read(controller / 'audit.json') or {}
        runtime = read(controller / 'container-runtime.json') or {}
        logical = (result.get('logical_query_cost') or {}).get('total')
        physical = (result.get('physical_evaluation_cost') or {}).get('total', {}).get('executions')
        code, worst, failed = official_codes(controller)
        note = f'failed={failed}; {worst}'
        print(f'| {repo} | {condition} | {result.get("passed")} | {code} | {logical} | {physical} | '
              f'{round(result.get("agent_wall_seconds") or 0, 1)} | {audit.get("model_steps")} | {note} |')
        print(f'  - agent_exit={runtime.get("agent_exit")} cleanup_error={runtime.get("cleanup_error")} '
              f'preflight_exit={runtime.get("preflight_exit")} audit_hits={len(audit.get("hits") or [])} '
              f'scope_violation={audit.get("scope_violation")} stop_reason={result.get("stop_reason")}')


if __name__ == '__main__':
    main()
