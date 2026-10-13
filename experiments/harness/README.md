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
still enforces filesystem, credential, and network isolation for formal experiments.

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
