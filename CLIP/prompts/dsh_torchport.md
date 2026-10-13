# DSH + TorchPort: frozen CLIP migration

You are the **DSH + TorchPort** method in a controlled migration experiment.
Migrate the official OpenAI CLIP ViT-B/32 in the current workspace from PyTorch
to native Jittor. Do not change the experimental protocol.

## Fixed scope

- Work only in `$CLIP_EXPERIMENT/workspaces/dsh_torchport/repo`.
- Implement `$CLIP_EXPERIMENT/workspaces/dsh_torchport/repo/submission.py`.
- `load_model(weights_path)` must return a `jittor.Module` loaded from the shared
  FP32 `weights_fp32.npz` archive.
- `forward_logits(model, images, tokens)` must return a `jittor.Var` of shape
  `[4, 4]`, containing CLIP image-to-text logits.
- Use the fixed `images.npy` and `tokens.npy`; preprocessing and tokenization are
  outside the comparison.
- No training, FP16, tokenizer migration, gradients, extra models, or extra
  inputs. TorchPort SA/FD may be recorded only as diagnostic evidence.

## Allowed TorchPort assistance

You may use the installed DSH TorchPort plugin and its structured tools:
`scan_repository`, `create_migration_plan`, `start_migration`,
`run_verification`, `locate_first_divergence`, `propose_repairs`,
`apply_verified_repair`, and `generate_certification_report`. You may also read
TorchPort output produced for this method. Do not copy code or results from the
Codex or pure DSH workspaces.

The final RN/NA/Max Error values must still come only from the common evaluator;
TorchPort certificates cannot replace them. Do not read or embed
`logits_ref.npy` in migrated code. Write only inside this method workspace and
`$CLIP_EXPERIMENT/results/dsh_torchport`.

## Evaluation and round counting

The initial implementation is round 0. Evaluate it with:

```bash
bash "$CLIP_EXPERIMENT/run_evaluate.sh" dsh_torchport
```

After a failed evaluation, and before each coherent repair, append exactly one
repair record (maximum 8):

```bash
"$SOURCE_PYTHON" "$CLIP_EXPERIMENT/scripts/record_round.py" \
  --method dsh_torchport \
  --decision "which TorchPort evidence supports this repair" \
  --candidate "the concrete general repair or code change" \
  --evidence "failure code, certificate fact, SA, or FD"
```

Then edit and evaluate again. Stop immediately when RN and NA are both true, or
when 8 repair rounds have been used. Keep TorchPort traces, but do not merge
SA/FD into the three-method result table. Do not fabricate or manually edit
result JSON files.
