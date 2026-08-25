# Codex baseline: frozen RAFT migration

You are the **Codex** baseline in a controlled migration experiment. Migrate the
official Princeton-VL RAFT at the frozen commit from PyTorch to native Jittor.
Do not change the experimental protocol.

## Fixed scope

- Work only in `$RAFT_EXPERIMENT/workspaces/codex/repo`.
- Implement `submission.py` using the common `load_model` / `forward_flow` contract.
- `load_model(weights_path)` must return a `jittor.Module` loaded from the shared
  FP32 `weights_fp32.npz` archive.
- `forward_flow(model, image1, image2)` must return `(flow_low, flow_up)` as
  `jittor.Var` values with shapes `[1,2,48,64]` and `[1,2,384,512]`.
- Preserve `iters=12`, `test_mode=True`, `small=False`,
  `mixed_precision=False`, and `alternate_corr=False`.
- Use only the frozen `image1.npy` and `image2.npy`; do not regenerate inputs.
- No training, benchmark evaluation, FP16, gradients, extra inputs, or RAFT-Stress.

## Isolation rules

You may read source code, edit this workspace, execute shell commands, run
PyTorch/Jittor, and inspect tracebacks. You must not:

- invoke any TorchPort command, module, plugin, API, certificate, SA, FD, or trace;
- read TorchPort source, `.torchport/`, or the `dsh_torchport` workspace;
- copy code or results from another method workspace;
- read or embed `flow_up_ref.npy` or `flow_low_ref.npy` in migrated code;
- modify `experiment.json`, `shared/`, evaluator scripts, or files outside this
  workspace and `$RAFT_EXPERIMENT/results/codex`.

## Evaluation and Rounds

The initial implementation is round 0. Evaluate only through:

```bash
bash "$RAFT_EXPERIMENT/run_evaluate.sh" codex
```

After a failed common evaluation, and before each coherent repair, append one
repair record (maximum 8):

```bash
"$SOURCE_PYTHON" "$RAFT_EXPERIMENT/scripts/record_round.py" \
  --method codex \
  --decision "why this repair follows from the evaluator failure" \
  --candidate "the concrete change" \
  --evidence "the previous evaluator error"
```

Stop when RN and NA are both true or after 8 repair rounds. Never fabricate or
manually edit evaluator result files.
