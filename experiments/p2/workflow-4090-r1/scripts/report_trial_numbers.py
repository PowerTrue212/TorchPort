"""Print the runbook step-5 numbers for one controller directory (numbers only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

controller = Path(sys.argv[1])
result = json.loads((controller / 'trial/result.json').read_text(encoding='utf-8'))
official = json.loads((controller / 'trial/official-verify.json').read_text(encoding='utf-8'))
events = [json.loads(line) for line in
          (controller / 'trial/events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]

print('trial_id', result.get('trial_id'))
print('stop_reason', result.get('stop_reason'), 'passed', result.get('passed'),
      'valid', result.get('valid'), 'claim', result.get('claim_status'))
print('agent_wall_seconds', round(result.get('agent_wall_seconds') or 0, 3))
print('total_wall_seconds', round(result.get('total_wall_seconds') or 0, 3))
print('official_wall_seconds', result.get('official_wall_seconds'))
print('tool_calls', result.get('tool_calls'), 'oracle_queries', result.get('oracle_queries'))
print('logical_query_cost', json.dumps(result.get('logical_query_cost')))
print('physical_evaluation_cost', json.dumps(result.get('physical_evaluation_cost')))
for name, report in official.items():
    print('official', name, 'passed', report.get('passed'), 'code', report.get('failure_code'),
          'e1_status', report.get('e1_status'))
    e1 = report.get('e1') or {}
    print('  e1 metrics', json.dumps(e1.get('metrics')))
    print('  first_divergence', json.dumps(e1.get('first_divergence')))
    runs = report.get('e1', {}).get('tensors') or []
    worst = max((t for t in runs if 'ratio' in t), key=lambda t: t['ratio'], default={})
    print('  worst_tensor', worst.get('name'), 'ratio', worst.get('ratio'))
print('events', len(events))
for event in events:
    if event.get('type') == 'tool_call' or 'tool' in event:
        print('  event', event.get('type'), event.get('tool'), 'ok', event.get('response_ok'),
              'error', event.get('error_type'))
