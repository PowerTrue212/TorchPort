# Jittor Harness

Jittor Harness controls reproducible migration trials for mature code Agents. Its implementation
stays in `experiments/harness/`; the internal validation package and CLI remain `torchport` for
compatibility. It does not call a
model API and does not replace an Agent's normal repository, Shell, Python, Git, test, or network
tools.

The Harness has four jobs:

1. Freeze the initial repository, task, feedback condition, budget, and hidden evaluator boundary.
2. Expose only TorchPort's validation tools to the Agent.
3. Record tool calls, logical queries, physical executions, candidate identities, and wall time.
4. Freeze the final repository and run hidden verification against that exact snapshot.

## Trial lifecycle

```text
complete source repository
        │
        ▼
independent Git candidate workspace
        │
        ├─ Agent checks environment, resources, data, and checkpoints
        ├─ Agent migrates repository code with its normal tools
        └─ Agent uses TorchPort feedback and iterates
        │
        ▼
repository finalizer
        ├─ repository snapshot
        ├─ candidate-manifest.json
        ├─ git-status.txt
        └─ candidate.diff
        │
        ▼
hidden verify on the frozen snapshot
```

The Agent decides how much preliminary checking to do. The Harness does not infer dependencies,
install packages, decide whether memory is sufficient, or download datasets. An external worker
still enforces filesystem and credential isolation for formal experiments. Public network access,
web search, and remote code search remain available with the same ordinary tool profile in every
feedback condition.

For every formal trial, the worker also keeps the Harness controller code, calibration
implementations, and every other candidate workspace outside the Agent's filesystem view. Each
condition starts a fresh Agent session with no previous trial history or persistent memory mounted,
and no access to another trial through session tools.
The worker records the ordinary Agent tool inventory and confirms that public network, web search,
and remote code search remain available. Public references used by the Agent are part of the trial,
and their URLs and revisions should be reported. The external integration retains web and code
search events separately from TorchPort oracle queries; finding an existing public port does not
invalidate the trial.
The controller requires these facts in the worker attestation before it freezes a formal run plan.
The JSON attestation is an audit record of worker-enforced isolation, not a sandbox by itself.
For `benchmark_protocol=benchmark-v1`, the run-plan validator additionally requires proof that
`import torchport` and forbidden-path read probes fail, that the Agent's remote shell cannot reach
the evaluator account or oracle, and that targeted answer/project URL filtering plus file/URL
access auditing are active. A DSH `workspace-write` setting alone limits writes, not all reads;
it does not satisfy this requirement.
If an Agent accesses a forbidden area or receives feedback from another condition, the worker
records `scope_violation`, `feedback_leak`, or `isolation_violation` as the stop reason. The Harness
keeps the final snapshot and raw evidence but excludes that attempt from valid and successful
trials. These stop reasons are not retried as infrastructure errors.

## Agent-facing tools

`HarnessGateway.tool_schemas()` returns only:

- `inspect`
- `run`
- `evaluate`
- `verify`
- `trace`
- `localize`

`trace` and `localize` are absent outside the localized condition. The Agent uses its own tools for
files, Shell, tests, Git, package management, and public research. `certify()` is controller-only and
is never part of the Agent tool schema.

An external integration creates one gateway for one Agent session:

```python
from experiments.harness import FeedbackLevel, HarnessGateway, TrialProtocol

gateway = HarnessGateway(
    task="public-task.json",
    hidden_task="hidden-task.json",
    evidence_root=".torchport/evidence",
    trial_dir="runs/plantcamo-none-r0",
    protocol=TrialProtocol(
        feedback=FeedbackLevel.DIAGNOSTIC,
        max_oracle_queries=10,
        max_wall_seconds=7200,
        candidate_entry="run.py",
    ),
)

tools = gateway.tool_schemas()
result = gateway.call("inspect")
```

The integration maps these schemas into the chosen mature Agent product. After the Agent session
ends, the trusted controller calls `gateway.certify(stop_reason)`. The Harness deliberately contains
no product-specific plugin and no direct model API client.

## Repository candidate contract

New trials start from a complete upstream repository copy. The Agent can change repository modules,
configuration, dependencies, tests, and the standard entry. A candidate is always a directory.

`run.py` is the default entry for current held-out tasks. It only imports and constructs the migrated
repository implementation. It must not contain the whole migrated model. Task-specific hidden tests
must also exercise the repository's normal imports, configuration, checkpoint loading, and inference
path; a file-count heuristic cannot prove a real repository migration.

`candidate.py` freezes the final tree, rejects common credential files, records every delivered file,
and saves Git review evidence when the workspace has a baseline commit. Official verification uses
the frozen copy, so edits made after finalization cannot affect the result.

## Directory responsibilities

| Path | Responsibility |
| --- | --- |
| `gateway.py` | TorchPort calls, feedback projection, budgets, event log, hidden certification |
| `candidate.py` | Repository manifest, clean snapshot, Git status and diff |
| `tooling.py` | Six stable TorchPort tool schemas and call validation |
| `feedback.py` | Verdict, diagnostic, and localized response projections |
| `orchestration/` | Trial matrix validation, frozen run plan, clean attempt copies and retry state |
| `reporting/` | Read-only aggregation and performance measurement |

The old DeepSeek/API Agent, restricted file tools, built-in web research, and model loop were removed.
Historical pilot evidence remains under `experiments/p2/` and is not rewritten as a current result.
