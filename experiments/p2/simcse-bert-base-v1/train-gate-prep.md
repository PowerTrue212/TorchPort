# SimCSE BERT-base training GATE preparation

Status: the source-side two-step checks are repeatable on the V100. A new
candidate seed contains the target training function names. The hidden
comparison and target tolerance are not frozen. No Agent training trial has
started. These checks do not turn the inference pilots into formal Benchmark
v1 results.

The full source-oracle preflight and its disk limit are in
[train-seed-preflight-r0-outcome.md](train-seed-preflight-r0-outcome.md).
The bounded observation preflight is in
[train-seed-preflight-r1-outcome.md](train-seed-preflight-r1-outcome.md).

## Fixed material

- Official SimCSE source: `49fa580a853752ede55b8c76d9debf748e214d3f`.
- BERT-base initialization file SHA-256:
  `097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a`.
- Wiki1M file SHA-256:
  `7b1825863a99aa76479b0456f7c210539dfaeeb69598b41fb4de4f524dd5a706`.
- Frozen batch: Wiki1M rows 0 through 7, official BERT tokenizer, maximum 32
  tokens, two identical tokenized views per sentence. Its manifest SHA-256 is
  `263820e4f71db594843680746f65d69514654dac1df28bd2406d4fbddab9a215`.
  Arrays have shape `[8, 2, 32]`; their individual hashes are in
  `D:\科研\SimCSE-Benchmark-v1\assets\train-batch-v1\manifest.json`.

The source check uses the upstream `BertForCL` implementation, `cls` training
pooler, temperature 0.05, FP32, seed 20261027, and two AdamW steps at learning
rate `3e-5`. It fixes AdamW epsilon at `1e-8`, beta at `(0.9, 0.999)`, and
weight decay at zero. It uses PyTorch 2.4.1 and Transformers 4.26.1 on a
16 GiB V100. The source loads the BERT encoder from the pinned checkpoint;
it initializes SimCSE's MLP from the fixed seed. The checkpoint's MLM and NSP
heads and BERT pooler are unused, as expected for this model.

This is a small diagnostic batch. The official example trains with batch 64,
one Wiki1M epoch, FP16, evaluation during training, and best-checkpoint
selection. The diagnostic check does not reproduce that complete procedure.

## Source results

| Dropout | Step 0 loss | Step 1 loss | Repeated run |
| --- | ---: | ---: | --- |
| 0 | 0.10187219083309174 | 0.000060228281654417515 | Bitwise equal |
| 0.1 | 0.266047865152359 | 0.01818563975393772 | Bitwise equal |

For each dropout setting, two fresh runs produced identical loss arrays,
similarity matrices, gradient norms, and SHA-256 digests of all model state
after each step. The `0.1` run exercises the stochastic augmentation used by
unsupervised SimCSE; its fixed seed made the two PyTorch runs repeatable. The
`0` run is the candidate for a direct numerical training comparison because
it does not require PyTorch and Jittor to generate identical dropout masks.

Raw source JSON/NPY/NPZ for all four runs are preserved in
`D:\科研\SimCSE-Benchmark-v1\controller\prep\train-source-evidence`. The
remote tar and downloaded tar both have SHA-256
`7adb93ca7d5f8d19a3c9c930f5cdb8cdd0eaa8c5dfd5a8633092107e223183af`.
`repeatability.json` records the independent comparison. Do not rewrite
these results when the target protocol changes.

The source check also saved four representative tensors: the first BERT
attention query weight, the last BERT output LayerNorm weight, and the SimCSE
MLP weight and bias. It saved their initial values, gradients, and values
after both steps. Two fresh FP32 runs produced bitwise equal arrays. The
complete initial `BertForCL` state has 200 items and 437,933,056 raw bytes.
Its NPZ SHA-256 is
`57a2f10c6a78ee6eb8f762a2620ff4f691eecc3841deb85d498b480675dc81c5`.
The source-state digest matches the archive. These files are in
`D:\科研\SimCSE-Benchmark-v1\controller\prep\train-source-tensors-evidence`.
The local and remote tar SHA-256 is
`ab5e9512c0a8eb45e627d7ad49f9a470d553a570c72881f4481fa34236ef9d0e`.

The standalone [source entry](train_source_entry.py) loads this initial state
and repeats both step losses and all post-step model-state digests exactly.
Its report is `source-entry-check.json` in the same evidence directory. The
entry uses [train_config.json](train_config.json), whose only changes from the
published BERT-base config are the two dropout probabilities, both set to 0.
The official source checkout has untracked copies of these two files. The
upstream commit remains unchanged.

One FP64 source run completed on the V100. Against the FP32 source run, the
maximum loss difference is `6.54e-8`, and the maximum 8-by-8 similarity
matrix difference is `3.91e-5`. The maximum difference across the selected
gradient tensors is `4.25e-8`. The raw FP64 JSON/NPY/NPZ and the comparison
are in `D:\科研\SimCSE-Benchmark-v1\controller\prep\train-source-fp64-evidence`.
The local and remote tar SHA-256 is
`57795b4f5777325ae5b755fa9b00fc405cf0254cd1b8aa0a6d2763d3f677c713`.
These source differences describe the numerical floor. They do not set a
target tolerance by themselves.

The two-step Harness inputs are in
`D:\科研\SimCSE-Benchmark-v1\assets\train-steps-v1`. Each array has shape
`[2, 8, 2, 32]`; both steps use the same frozen batch. The manifest SHA-256
is `11eca3e3b808194599e30b069453ae962857f412b49d30e6a973cfec72ce445a`.

The next candidate seed is
`D:\科研\SimCSE-Benchmark-v1\seed-train\SimCSE` at commit
`1d20672a558eae18127c0ae84ebb3566fc716121`. Its `run.py` declares
`load_train_model`, `build_optimizer`, and `train_step`, next to the existing
inference entry. The candidate training model must expose the same 200 unique
state names and shapes, including the position-ID buffer. The optimizer must
update all trainable model parameters. A target implementation is still
absent, as intended for the Agent task.

## Work before the next Agent trial

1. Freeze the target TaskSpec against the new seed and its 200-key training
   initialization. Preserve the published inference checkpoint separately.
2. Choose and validate target tolerances on the V100. Keep a failed source or
   target execution as failure. Do not infer parity from source repeatability.
3. Validate the bounded observation TaskSpec against a correct native Jittor
   training reference and negative controls. The source preflight works; the
   target tolerance remains exploratory. State the reduced numerical coverage.
4. Create `none` and `full` checkouts from the new seed commit. Freeze their
   common prompt, assets, container, budget, and hidden task before starting
   either Agent.

The current `none-r0` and `full-r0` remain inference-only pilots. Their
scores and validity are recorded in [pilot-r0-comparison.md](pilot-r0-comparison.md).
