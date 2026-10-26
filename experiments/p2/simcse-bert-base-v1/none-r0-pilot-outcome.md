# SimCSE none-r0 inference pilot outcome

Date: 2026-10-27. This is a `pilot_only` run. It is outside the frozen
Benchmark v1 repository matrix. The training check is not frozen.

The Agent completed with exit code 0 after 79 model steps. The controller
froze a complete 65-file candidate repository. The original result at
`D:\科研\SimCSE-Benchmark-v1\controller\simcse-none-r0\trial\result.json`
reports `passed=true`, `valid=false`, `stop_reason=agent_completed`, and
`claim_status=pilot_only`. The trial is not a formal benchmark result.

The hidden full STS-B verify passed twice. Both runs loaded all 200 checkpoint
state items with no missing, extra, or mismatched values. The output structure
and FP32 numerical check passed. The reported maximum absolute output error
was `9.834766387939453e-6`, with zero FP32 mismatches.

The frozen task scorer used 1,379 STS-B test pairs. Both target repeats and
the source had Spearman correlation `0.7685349386294106`. Under the pilot
tolerance of `0.005`, both repeat scores give `P_t=100`. This is an inference
task score. It does not establish training parity or formal trial validity.

The Agent used two logical oracle queries: one `evaluate` and one `verify`.
They covered one unique candidate. The controller recorded eight Agent-side
physical executions and four hidden verification executions. These counts
include source and target work as defined by the Harness. The Agent wall time
was 2,324.75 seconds. The final verification took 116.51 seconds.

The original audit found no canary, forbidden-path, or forbidden-URL hit.
It recorded two incomplete JSONL lines and listed text that looks like paths
outside `/work` for manual review. The corrected scan in `audit-rescan.json`
also found zero hits. The preflight had found the protected host paths
inaccessible. These checks support the pilot audit; they do not prove complete
isolation from public information available through normal network search.

Raw evidence is in `D:\科研\SimCSE-Benchmark-v1\controller\simcse-none-r0`:

- `agent.jsonl`, `container-runtime.json`, `audit.json`, and `audit-rescan.json`.
- `trial\delivery\repository`, `trial\official-verify.json`, and `trial\result.json`.
- `task-score\score.json` and `task-score-repeat1\score.json`, with cosine NPZ files.
- Verified local tar archives of remote development and hidden verification
  evidence: `remote-evidence-f970d1567142.tar` and
  `remote-evidence-667562139567.tar`. Each has a `.manifest.json` with the
  source file hashes. The controller removed the archived remote files only
  after it verified both tar archives against the remote hashes.
