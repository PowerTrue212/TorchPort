# SimCSE full-r0 inference pilot outcome

Date: 2026-10-27. This is a `pilot_only` run. It is outside the frozen
Benchmark v1 repository matrix. The training GATE is not frozen.

## Frozen delivery and technical result

The Agent edited the `full` checkout. The controller froze the complete
repository at `D:\科研\SimCSE-Benchmark-v1\controller\simcse-full-r0\trial\delivery\repository`.
The original controller result is `trial\result.json`. It reports
`passed=false`, `valid=false`, `stop_reason=agent_error`, and
`claim_status=pilot_only`. The Agent exited with code 1 after 105 model steps.
The audit found zero canary, forbidden-path, or forbidden-URL hits. Its final
JSONL line is incomplete because the Agent process stopped during a tool call.

The hidden STS-B verify ran on the frozen snapshot. It reports
`WEIGHT_LOAD_FAILED`: 199 of 200 checkpoint state items loaded. The only
missing item is `model.embeddings.position_ids`. Output structure and the
frozen FP32 numerical check passed. The FP32 comparison covered 2,647,680
elements. It found zero mismatches and a maximum absolute error of
`9.834766387939453e-6`. The stricter diagnostic comparison failed on 2,043
elements; strict alignment is not the acceptance result.

The separate STS-B scorer used the frozen target embeddings and 1,379 official
test pairs. Source and target Spearman correlations were both
`0.7685349386294106`. The declared pilot tolerance was `0.005`, so the task
score calculation gave `P_t=100`. This is a task-metric observation. It does
not override the checkpoint-loading failure or make the trial formally valid.

## Infrastructure and recovery

The first controller process exited before audit and certification. Windows
decoded Docker's output with GBK and raised `UnicodeDecodeError` while cleaning
the container. All Agent logs remained in the controller directory. The
controller-only audit was run afterward. Its original `audit.json` and a
separate `audit-rescan.json` are both retained.

The first hidden verification attempt then failed because the remote data
filesystem rejected a new file with `ENOSPC`. Its raw stdout, stderr, and exit
code are in `trial\remote-raw\0001-verify.json`. The controller removed 7,518
regenerable `.pyc` files from the source Python environment; it did not remove
JSON, JSONL, NPZ, model weights, or trial evidence. A write probe then passed.
The controller retried certification against the unchanged frozen snapshot.
It saved the final `trial\official-verify.json` and `trial\result.json`.

The repeated Agent-visible remote `run` calls before Agent exit returned a
generic service error. The previous executor did not save raw stdout or stderr
for a non-JSON remote exit. Their exact causes cannot be reconstructed. The
executor now saves raw failed executions in the trusted controller, without
exposing private logs to the Agent.

## Evidence

- Original Agent log: `D:\科研\SimCSE-Benchmark-v1\controller\simcse-full-r0\agent.jsonl`
- Container runtime: `D:\科研\SimCSE-Benchmark-v1\controller\simcse-full-r0\container-runtime.json`
- Audit: `D:\科研\SimCSE-Benchmark-v1\controller\simcse-full-r0\audit.json`
- Frozen delivery and official result: `D:\科研\SimCSE-Benchmark-v1\controller\simcse-full-r0\trial\`
- Independent frozen verification and STS-B score:
  `D:\科研\SimCSE-Benchmark-v1\controller\simcse-full-r0\diagnostics\simcse-full-r0-frozen-verify-r0\`
- Verified remote evidence archive:
  `D:\科研\SimCSE-Benchmark-v1\controller\simcse-full-r0\remote-evidence-c9be255ff1e2.tar`
  and its `.manifest.json`. The remote files were removed only after the
  archive matched all 34 remote file hashes. The seed STS-B source evidence
  was also archived under `controller\prep\simcse-seed-stsb-r2` before its
  remote copies were removed.

The `none` controller started from the same clean seed commit after this
full-r0 result was recorded. Its separate outcome is in
[none-r0-pilot-outcome.md](none-r0-pilot-outcome.md).
