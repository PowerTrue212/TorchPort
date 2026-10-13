# Codex baseline: frozen CLIP migration

You are the **Codex** baseline in a controlled migration experiment. Migrate the
official OpenAI CLIP ViT-B/32 in the current workspace from PyTorch to native
Jittor. Do not change the experimental protocol.

## Fixed scope

- Work only in `$CLIP_EXPERIMENT/workspaces/codex/repo`.
- Implement `$CLIP_EXPERIMENT/workspaces/codex/repo/submission.py`.
- `load_model(weights_path)` must return a `jittor.Module` loaded from the shared
  FP32 `weights_fp32.npz` archive.
- `forward_logits(model, images, tokens)` must return a `jittor.Var` of shape
  `[4, 4]`, containing CLIP image-to-text logits.
- Use the fixed `images.npy` and `tokens.npy`; do not implement preprocessing or
  tokenization.
- No training, FP16, tokenizer migration, gradients, layer-wise metrics, extra
  models, or extra inputs.

## Isolation rules

You may read source code, edit this workspace, and run shell/PyTorch/Jittor.
You must not:

- invoke any `torchport` command, module, plugin, API, or generated report;
- read TorchPort source code, `.torchport/`, certificates, SA/FD traces, or the
  `dsh_torchport` workspace;
- copy code or results from either other method workspace;
- read or embed `logits_ref.npy` in migrated code;
- write outside this method workspace and `$CLIP_EXPERIMENT/results/codex`.

## Evaluation and round counting

The initial implementation is round 0. Evaluate it with:

```bash
bash "$CLIP_EXPERIMENT/run_evaluate.sh" codex
```

After a failed evaluation, and before each coherent repair, append exactly one
repair record (maximum 8):

```bash
"$SOURCE_PYTHON" "$CLIP_EXPERIMENT/scripts/record_round.py" \
  --method codex \
  --decision "why this repair follows from the failure" \
  --candidate "the concrete change" \
  --evidence "the previous evaluator error"
```

Then edit and evaluate again. Stop immediately when RN and NA are both true, or
when 8 repair rounds have been used. Do not fabricate or manually edit result
JSON files.
