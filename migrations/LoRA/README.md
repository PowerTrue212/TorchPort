# Microsoft LoRA paired case

This directory prepares a PyTorch-to-Jittor paired case around GPT-2 Small and LoRA. Third-party
repositories, checkpoints, generated workspaces, and run outputs stay local and are ignored by Git.

Frozen upstream revisions:

- PyTorch source: `microsoft/LoRA@c4593f060e6a368d7bb5af5273b8e42810cdef90`
- V100 paired reference: `zhenrys/LoRA-GPT2-E2E-pytorch-jittor@8ee1d5064879a26e7ad9eddcb7bfbb46e27ad253`
- secondary calibration reference: `weihuakuang/LoRA-Jittor@8920778156e43c7ff6851d731ed75525a34b5c99`

Expected local layout:

```text
migrations/LoRA/
├── source/microsoft-LoRA/
├── reference/LoRA-GPT2-E2E-pytorch-jittor/
├── reference/LoRA-Jittor/
└── checkpoint/gpt2-pytorch_model.bin
```

The intended initial scope is GPT-2 Small (`gpt2.sm`: 12 layers, hidden size 768, 12 attention
heads), not the medium model used in the published E2E command. The source training path exposes
three useful migration boundaries: freezing all non-LoRA parameters, injecting LoRA into the fused
QKV projection, and saving/loading LoRA checkpoints separately from the pretrained GPT-2 weights.

The downloaded `gpt2-pytorch_model.bin` is the pretrained GPT-2 Small/base weight file referenced by
the official repository. It is not a trained LoRA checkpoint. The official release only publishes
task LoRA checkpoints for GPT-2 Medium and Large, so a Small LoRA checkpoint must be produced by a
controlled training run or created as a synthetic verification fixture.

The V100 reference keeps separate `by-pytorch/` and `by-jittor/` implementations and records a
GPT-2 Small run. Its README reports the PyTorch training run on a 32GB V100, but the Jittor run on a
24GB RTX 3090. It is paired code, not same-hardware paired evidence; a fair benchmark must rerun both
sides on the same V100. The secondary reference covers three NLG datasets and has a cleaner direct
`src/` versus `src_jittor/` layout, but its own reported runs also used an RTX 3090. Treat both as
calibration evidence, not candidate input for a held-out trial.

## Local preparation status

Prepared on 2026-10-17:

- The fixed source archive and both fixed reference archives were downloaded and their SHA-256
  values match `upstreams.json`.
- The official GPT-2 Small/base checkpoint is present and matches the recorded size and SHA-256.
  It was not deserialized during preparation.
- The Python files that implement GPT-2 training, LoRA injection, parameter freezing, and checkpoint
  filtering compile under the project's Python 3.12 environment.
- The V100 reference's `by-jittor/NLG/src/gpt2_beam.py` does not compile at the frozen revision:
  line 257 contains `else` without a colon. This is outside the initial training/forward/checkpoint
  scope, but blocks treating the repository as an already-runnable end-to-end inference reference.
- No GPU result has been produced locally. The current machine is not a V100, and the configured
  remote GPU was unavailable during preparation.

Run `powershell -File migrations/LoRA/verify_materials.ps1` to recheck the local downloads without
loading the checkpoint. Do not install the official `torch==1.7.1+cu101` requirements into the main
TorchPort environment; use isolated source and Jittor environments when runtime work begins.

Before turning this into a benchmark, freeze the exact input batch, pretrained state, LoRA state,
train/eval mode, random seeds, dtype, device, and checkpoint conversion rules in a TaskSpec. Keep the
published training and task metrics separate from forward/checkpoint equivalence. `verify` remains
the only final acceptance decision.

The paired `none` / `full` candidate states and framework-neutral fixtures are prepared under
`experiments/p2/lora-gpt2-small-v1/`. Their current status is `prepared_not_runnable`: the two local
workspaces have identical candidate code, but source/target adapters, TaskSpec, oracle, same-V100
runtime validation, and the current external Agent tool mapping still need to be completed before a
formal trial.

## 2026-10-24 update

The original note above describes the initial 2026-10-17 preparation, not the current pilot state.
The local checkpoint directory now contains the verified official GPT-2 Small base weights
(`gpt2-pytorch_model.bin`, SHA-256 `7c5d3f4b...`) and a 703 MB calibration NPZ assembled from those
weights plus a deterministic LoRA fixture (`gpt2-sm-lora-train.npz`, SHA-256 `130b2a52...`). The
calibration NPZ is also cached on the V100. Neither file is a task-trained GPT-2 Small E2E LoRA
checkpoint; no such checkpoint was found in the current V100 experiment directories. The raw E2E
train/valid/test text is already local, but its encoded JSONL and task-generation metrics are not
prepared. See the [V100 pilot record](../../experiments/p2/lora-gpt2-small-v1/train/v100-calibration-20261024.md).

A fresh r1 none/full pair uses a neutral prompt and evaluation entry in separate candidate workspaces.
It is an open-tool pilot, not a claim of host filesystem isolation or a completed Benchmark v1 trial.
Its paths and hashes are recorded in
[`prepared-state-r1.json`](../../experiments/p2/lora-gpt2-small-v1/prepared-state-r1.json).
