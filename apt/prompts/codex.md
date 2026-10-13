# Codex baseline: frozen APT migration

You are the **Codex** baseline in a controlled migration experiment. Migrate the
complete APT CIFAR-100 training and inference workflow in the current workspace
from PyTorch to native Jittor. Do not change the protocol.

## Fixed scope

- Work only in `$APT_EXPERIMENT/workspaces/codex/repo`.
- Implement `submission.py::run_experiment`; supporting Jittor modules may be
  added inside this workspace.
- Train all 10 class-incremental tasks, 30 epochs per task, then run the final
  model over all 10,000 CIFAR-100 test images.
- Use only the supplied CIFAR-100 data and ViT-B/16 checkpoint.
- Preserve class order, preprocessing, seed, batch size, Adam settings, APT
  prompt behavior, EMA merge, and FP32 precision from `experiment.json`.

## Isolation rules

You may read the original source and edit this workspace. You must not:

- invoke any `torchport` command, module, plugin, API, report, trace, SA, or FD;
- read TorchPort source code or the `dsh_torchport` workspace;
- copy code or results from either other method workspace;
- import or execute PyTorch from the final submission;
- read the PyTorch reference results or prediction arrays;
- write outside this workspace and `$APT_EXPERIMENT/results/codex`.

## Evaluation and repair rounds

The initial implementation is round 0. Evaluate with:

```bash
bash "$APT_EXPERIMENT/run_evaluate.sh" codex
```

After a failed evaluation, and before each coherent repair, record exactly one
round (maximum 8):

```bash
"$SOURCE_PYTHON" "$APT_EXPERIMENT/scripts/record_round.py" \
  --method codex \
  --decision "why this repair follows from the failure" \
  --candidate "the concrete change" \
  --evidence "the previous evaluator failure"
```

Stop when RN is true or 8 repair rounds have been used. Average Accuracy,
Forgetting, and Delta Accuracy come only from the common evaluator; do not
fabricate result files.
