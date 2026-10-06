# Controlled Agent Harness

This directory is the experiment boundary around TorchPort. It is deliberately outside
`src/torchport`: TorchPort judges evidence, while this harness controls what an Agent can see and
counts the cost of obtaining that evidence.

## Conditions

- `verdict`: `evaluate` and `verify` return only `passed`.
- `diagnostic`: a fixed allowlist of stage/category/code, aggregate numeric errors, structural,
  weight-load, aligned, and strict-aligned status. It never returns free text or a location.
- `localized`: the full diagnostic result plus `trace` and `localize` is available.

All conditions use the same TorchPort evaluator. Only the response shown to the Agent changes.

The diagnostic allowlist excludes module, file, line, operator, tensor name, first divergence,
`summary`, `problems`, and `next`. Localized feedback adds scope, invocation, operator, first
divergence, available input/output evidence, and error jump.

Logical query cost and physical evaluation cost are recorded independently. `evaluate`, `verify`,
and `trace` consume logical queries. `probe` and `run` do not consume logical queries, but every
process they start contributes to physical cost. Physical cost records process executions, their
wall seconds, and wall seconds for CUDA-configured executions. The CUDA value is an allocation-time
upper bound, not measured GPU active time.

## Trusted boundary

The model receives `HarnessGateway.tool_schemas()` and can invoke only `HarnessGateway.call()`.
The registered tools can read the source repository, can read and edit the target repository, plus
`inspect`, `probe`, `run`, `evaluate`, `verify`, `trace`, and `localize`. There is no generic shell
tool.
`apply_patch` is always target-only; passing a source side or escaping either repository is rejected.

`read_file` returns at most 200 lines and reports the returned range, total line count, and whether
more lines remain. Agents should use `search_code` first and request only the ranges they need. This
keeps repeated provider context from growing with whole repository files.

`probe(path, args=[])` runs one existing `.py` file inside the target repository. The target Python,
device, environment, cache root, working directory, and timeout come from the frozen TaskSpec and
trial; the Agent cannot override them. Arguments are passed as an argv array without a shell. Each
call gets its own stdout/stderr logs and `probe.json`; the response returns only the last 20,000
characters of each stream. Credential-like variables inherited by the controller are removed before
the explicit TaskSpec environment is applied. A probe result is candidate-owned evidence and cannot
certify success; the Gateway never passes oracle data or paths to it.

The Gateway controls which process is launched and what evidence is returned. It is not an operating
system sandbox: a formal benchmark must run the whole trial in a container or similarly isolated
worker if candidate code itself is untrusted or network/filesystem isolation is required.

`HarnessGateway.certify()` is intentionally absent from the tool schemas. The trusted trial
controller calls it after the Agent loop returns. A hidden TaskSpec may use different inputs and
source evidence, but it must point at the same target workspace.

```python
from experiments.harness import FeedbackLevel, HarnessGateway, TrialProtocol, run_trial

gateway = HarnessGateway(
    task="benchmarks/tasks/tiny/task.yaml",
    evidence_root=".torchport/p2/tiny/t0",
    trial_dir="experiments/p2/runs/pilot/tiny/verdict/t0",
    trial_id="t0",
    protocol=TrialProtocol(
        feedback=FeedbackLevel.VERDICT,
        max_oracle_queries=10,
        max_wall_seconds=3600,
    ),
)

def agent(tool_schemas, call):
    # A provider adapter sends only tool_schemas to the model and maps tool calls to call().
    call("inspect")
    return "agent_completed"

result = run_trial(gateway, agent)
```

Every response is saved under `tool-results/` and referenced by hash from `events.jsonl`.
A trusted provider adapter should record raw token usage with `gateway.record_agent_event()`.

## DeepSeek API

Copy the example beside the harness and fill only the key. Keeping this file out of the repository
root lets `dsh web` start normally; DSH rejects launch-control variables from a workspace `.env`.

```powershell
Copy-Item experiments/harness/.env.example experiments/harness/.env
notepad experiments/harness/.env
```

```dotenv
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

Check the configuration without sending a request:

```powershell
.venv\Scripts\python.exe -m experiments.harness.deepseek --check
```

Run a controlled trial:

```powershell
.venv\Scripts\python.exe -m experiments.harness.deepseek `
  --env experiments/harness/.env `
  --task benchmarks/tasks/tiny/task.yaml `
  --prompt-file experiments/harness/prompt.example.md `
  --feedback diagnostic `
  --query-budget 10 `
  --wall-seconds 3600 `
  --trial-id tiny-diagnostic-0
```

For a pre-registered comparison, use the experiment controller instead of invoking trials one by
one. It validates the complete task/condition/replicate matrix, verifies frozen artifact hashes,
captures interpreter/package/GPU facts, and writes immutable `run-plan.json` and
`runtime-snapshot.json` files before the first provider request:

```powershell
.venv\Scripts\python.exe -m experiments.harness.experiment `
  experiments/p2/<experiment>/experiment.json `
  --env experiments/harness/.env `
  --allow-trusted-pilot `
  --prepare-only

.venv\Scripts\python.exe -m experiments.harness.experiment `
  experiments/p2/<experiment>/experiment.json `
  --env experiments/harness/.env `
  --allow-trusted-pilot
```

`--allow-trusted-pilot` is required when the manifest has
`isolation.formal_claims_allowed=false`. Such a run is useful for infrastructure validation but
cannot support a formal isolation claim.

Add task-specific Python distributions to the manifest's `runtime_packages` list. The runtime
snapshot records their installed versions, or their absence, for both source and target
interpreters in addition to TorchPort's core numeric packages. This is audit metadata; dependency
installation remains an explicit worker setup step.

Each planned trial is one denominator unit. The controller copies the frozen seed candidate into a
new attempt directory, records progress atomically in `run-state.json`, and resumes completed
trials. An infrastructure failure may be retried once by default; the retry starts from a fresh seed
and remains an attempt of the same planned trial. The aggregate selects the first valid attempt, so
retries never enlarge the Success@Budget denominator. Model failure, budget exhaustion, and a valid
failed official verification are not retried.

Formal runs require an attestation produced by the isolated worker:

```powershell
.venv\Scripts\python.exe -m experiments.harness.experiment `
  experiments/p2/<experiment>/experiment.json `
  --env experiments/harness/.env `
  --worker-attestation path/to/worker-attestation.json
```

The example at `worker-attestation.example.json` deliberately contains `false` placeholders and
cannot pass the gate accidentally. The worker must attest that the network and host filesystem are
isolated, the candidate is the only writable mount, source and frozen inputs are read-only, and
hidden artifacts are absent from the Agent mount. It must also record an immutable image digest.
The JSON file is evidence and a controller gate; it does not create a sandbox. A container or other
external supervisor must enforce these properties before emitting the attestation.

Freeze the experiment id, replicate count, budgets, model, prompt, and sampling settings before a
comparison. The controller writes `summary.json` automatically. The aggregate can also be rebuilt
from the frozen plan without provider access:

```powershell
.venv\Scripts\python.exe -m experiments.harness.aggregate `
  experiments/p2/tiny-smoke/run-plan.json `
  --output experiments/p2/tiny-smoke/summary.json
```

Measure performance only after official verification has passed. This reuses normal capture,
disables trace, and checks correctness after the benchmark loop:

```powershell
.venv\Scripts\python.exe -m experiments.harness.performance `
  task.yaml path/to/official-verify.json `
  --root .torchport/performance `
  --trial performance-r0 `
  --warmup 10 --iterations 30 `
  --output performance.json
```

For a formal benchmark, also pass `--hidden-task` pointing to an immutable TaskSpec that uses the
same target repository. The adapter uses non-thinking Chat Completions tool calls and records the
provider's input, cached-input, output, reasoning, and total token classes in `events.jsonl`.
