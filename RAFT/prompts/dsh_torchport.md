# DSH + TorchPort: frozen RAFT migration

You are the **DSH + TorchPort** method in a controlled migration experiment.
Migrate the official Princeton-VL RAFT at the frozen commit from PyTorch to
native Jittor without changing the protocol.

## Fixed scope

- Work only in `$RAFT_EXPERIMENT/workspaces/dsh_torchport/repo`.
- Implement `submission.py` using the common `load_model` / `forward_flow` contract.
- Load the shared FP32 `weights_fp32.npz` and return a native `jittor.Module`.
- Return `(flow_low, flow_up)` as Jittor variables with shapes
  `[1,2,48,64]` and `[1,2,384,512]`.
- Preserve `iters=12`, `test_mode=True`, `small=False`,
  `mixed_precision=False`, and `alternate_corr=False`.
- Use only the frozen fixture. No training, benchmark evaluation, FP16,
  gradients, extra inputs, or RAFT-Stress.

## Allowed TorchPort assistance

You may use the installed DSH TorchPort plugin and these structured tools:
`scan_repository`, `create_migration_plan`, `start_migration`,
`run_verification`, `locate_first_divergence`, `propose_repairs`, and
`apply_verified_repair`. Keep certificate and first-divergence evidence only in
this method's result/trace directory.

TorchPort evidence cannot replace the common evaluator. Do not copy code or
results from Codex or pure DSH, do not embed frozen reference arrays, and do not
modify `experiment.json`, `shared/`, or evaluator scripts.

## Evaluation and Rounds

The initial implementation is round 0:

```bash
bash "$RAFT_EXPERIMENT/run_evaluate.sh" dsh_torchport
```

After a failed common evaluation, record one repair before editing:

```bash
"$SOURCE_PYTHON" "$RAFT_EXPERIMENT/scripts/record_round.py" \
  --method dsh_torchport \
  --decision "which TorchPort evidence supports this repair" \
  --candidate "the concrete repair" \
  --evidence "failure, certificate fact, SA, or first divergence"
```

Stop when RN and NA are both true or after 8 repair rounds. Preserve TorchPort
diagnostics separately from the RN/NA result table.
