# DSH + TorchPort: frozen APT migration

You are the **DSH + TorchPort** method in a controlled migration experiment.
Migrate the complete APT CIFAR-100 training and inference workflow in the
current workspace from PyTorch to native Jittor. Do not change the protocol.

## Fixed scope

- Work only in `$APT_EXPERIMENT/workspaces/dsh_torchport/repo`.
- Implement `submission.py::run_experiment`; supporting Jittor modules may be
  added inside this workspace.
- Train all 10 class-incremental tasks, 30 epochs per task, then run the final
  model over all 10,000 CIFAR-100 test images.
- Use only the supplied CIFAR-100 data and ViT-B/16 checkpoint.
- Preserve class order, preprocessing, seed, batch size, Adam settings, APT
  prompt behavior, EMA merge, and FP32 precision from `experiment.json`.

## TorchPort assistance

You may use the installed DSH TorchPort plugin and its structured tools:
`scan_repository`, `create_migration_plan`, `start_migration`,
`run_verification`, `locate_first_divergence`, `propose_repairs`,
`apply_verified_repair`, and `generate_certification_report`. You may read
TorchPort evidence produced only for this method. Do not copy from Codex or pure
DSH workspaces.

The common evaluator remains the only source of RN, Average Accuracy,
Forgetting, and Delta Accuracy. TorchPort SA/FD are diagnostic evidence only.
Do not import PyTorch in the final submission or read the PyTorch reference
results/predictions.

## Evaluation and repair rounds

The initial implementation is round 0. Evaluate with:

```bash
bash "$APT_EXPERIMENT/run_evaluate.sh" dsh_torchport
```

After a failed evaluation, and before each coherent repair, record exactly one
round (maximum 8):

```bash
"$SOURCE_PYTHON" "$APT_EXPERIMENT/scripts/record_round.py" \
  --method dsh_torchport \
  --decision "which TorchPort evidence supports this repair" \
  --candidate "the concrete change" \
  --evidence "failure code, certificate fact, SA, or FD"
```

Stop when RN is true or 8 repair rounds have been used. Keep SA/FD traces out of
the main three-method table and never fabricate result files.
