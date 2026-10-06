# CamoFormer V4.1 Flash final pilot evidence

This directory contains the compact trusted-pilot evidence described in
`deliverables/camoformer-v41-agent-pilot-report-2026-10-04.md`.

- `agent-pilot-compact.tgz`: immutable downloaded archive, SHA256
  `35fd45d373214918162176d3b77358b63b2c1a245e32b62c630dffff93d2f85f`.
- `agent-pilot/`: safely extracted manifest, plan, runtime snapshot, aggregate, run state, events,
  selected results, and official verification reports.
- `provider-model-snapshot.json`: live `/models` resolution of API ID `deepseek-flash` to
  `DeepSeek-V4.1-Flash`.

All three planned trials are valid failures with `token_budget_exhausted`; Success@Budget is 0/3.
The Agent made no candidate revision and issued no logical verification query. Official hidden
certification still ran and is reported separately from Agent cost.
