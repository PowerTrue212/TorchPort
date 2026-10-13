# Pure DSH baseline: frozen RAFT migration

You are the **pure DSH** baseline in a controlled migration experiment. Migrate
the official Princeton-VL RAFT at the frozen commit from PyTorch to native
Jittor. Do not change the experimental protocol.

## Fixed scope

- Work only in `$RAFT_EXPERIMENT/workspaces/dsh/repo`.
- Implement `submission.py` using the common `load_model` / `forward_flow` contract.
- Load the shared FP32 `weights_fp32.npz` and return a native `jittor.Module`.
- Return `(flow_low, flow_up)` as Jittor variables with shapes
  `[1,2,48,64]` and `[1,2,384,512]`.
- Preserve `iters=12`, `test_mode=True`, `small=False`,
  `mixed_precision=False`, and `alternate_corr=False`.
- Use only the frozen `image1.npy` and `image2.npy`; do not regenerate inputs.
- No training, benchmark evaluation, FP16, gradients, extra inputs, or RAFT-Stress.

## Isolation rules

This session is valid only when it was started by the experiment-owned launcher:

```powershell
.\run_dsh_pure.ps1
```

The launcher disables the installed TorchPort plugin in DSH's composed runtime
configuration, removes TorchPort environment variables from the child process,
and writes `results/dsh/isolation/latest.json`. Stop immediately if its preflight
audit does not report `passed=true` and `active_torchport_plugin=false`.

DSH may read the repository, edit its own workspace, execute shell commands,
run PyTorch/Jittor, and use ordinary traceback-driven repair. It must not:

- invoke `scan_repository`, `start_migration`, `run_verification`,
  `locate_first_divergence`, or any other TorchPort capability;
- read TorchPort source, certificates, SA/FD evidence, traces, or the
  `dsh_torchport` workspace;
- copy code or results from another method workspace;
- read or embed the frozen reference arrays in migrated code;
- modify `experiment.json`, `shared/`, evaluator scripts, or files outside this
  workspace and `$RAFT_EXPERIMENT/results/dsh`.

## Evaluation and Rounds

The initial implementation is round 0:

```bash
bash "$RAFT_EXPERIMENT/run_evaluate.sh" dsh
```

After a failed common evaluation, record one repair before editing:

```bash
"$SOURCE_PYTHON" "$RAFT_EXPERIMENT/scripts/record_round.py" \
  --method dsh \
  --decision "why this repair follows from the evaluator failure" \
  --candidate "the concrete change" \
  --evidence "the previous evaluator error"
```

Stop when RN and NA are both true or after 8 repair rounds. Never fabricate or
manually edit evaluator result files.
