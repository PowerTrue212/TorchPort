"""Print the workflow r3 trial table (numbers only) from the three controller directories."""
from __future__ import annotations

import json
from pathlib import Path

TRIALS = [
    ('RAFT 推理+训练', 'none', Path('D:/科研/RAFT-Benchmark-v1/controller/raft-wf-none-r3')),
    ('RAFT 推理+训练', 'full', Path('D:/科研/RAFT-Benchmark-v1/controller/raft-wf-full-r3')),
    ('FNO 推理', 'none', Path('D:/科研/FNO-Benchmark-v1/controller/fno-wf-none-r6')),
]


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def official(controller: Path) -> str:
    report = read(controller / 'trial' / 'official-verify.json') or {}
    parts = []
    for name, value in report.items():
        e1 = (value.get('e1') or {}).get('metrics') or {}
        result = (value.get('results') or [{}])[0]
        env = result.get('envelope_metrics') or {}
        ratio = e1.get('max_ratio') if e1 else env.get('max_gating_usage')
        marginal = (value.get('marginal_rerun') or {}).get('median_usage')
        parts.append(f'{name}:{value.get("failure_code") or "PASS"} ratio/usage={ratio}'
                     + (f' marginal_median={marginal}' if marginal is not None else ''))
    return '; '.join(parts)


def main() -> None:
    print('| 仓库 | 组 | passed | 官方判定 | logical query（evaluate/verify/trace） | physical 执行 | agent 墙钟(s) | 模型步数 |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for repo, condition, controller in TRIALS:
        result = read(controller / 'trial' / 'result.json')
        if result is None:
            print(f'| {repo} | {condition} | 未完成 | - | - | - | - | - |')
            continue
        audit = read(controller / 'audit.json') or {}
        runtime = read(controller / 'container-runtime.json') or {}
        logical = result.get('logical_query_cost') or {}
        by_tool = logical.get('by_tool') or {}
        physical = ((result.get('physical_evaluation_cost') or {}).get('total') or {}).get('executions')
        print(f'| {repo} | {condition} | {result.get("passed")} | {official(controller)} | '
              f'{logical.get("total")}（{by_tool.get("evaluate")}/{by_tool.get("verify")}/{by_tool.get("trace")}） | '
              f'{physical} | {round(result.get("agent_wall_seconds") or 0, 1)} | {audit.get("model_steps")} |')
        print(f'  - agent_exit={runtime.get("agent_exit")} cleanup_error={runtime.get("cleanup_error")} '
              f'preflight_exit={runtime.get("preflight_exit")} audit_hits={len(audit.get("hits") or [])} '
              f'scope_violation={audit.get("scope_violation")} stop_reason={result.get("stop_reason")}')


if __name__ == '__main__':
    main()
