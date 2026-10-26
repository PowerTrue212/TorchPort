# SimCSE BERT-base pilot preparation

Status: both inference pilots finished on 2026-10-27. The
[comparison](pilot-r0-comparison.md) records a passed hidden inference check
for `none-r0` and a checkpoint-loading failure for `full-r0`. The technical
STS-B scores matched the source in both candidates. Both runs remain
`pilot_only`. Training parity and formal benchmark validity are not shown.
The source-side training checks and remaining GATE work are in
[train-gate-prep.md](train-gate-prep.md).

## Fixed upstream material

| Item | Revision |
| --- | --- |
| Princeton NLP SimCSE source | `49fa580a853752ede55b8c76d9debf748e214d3f` |
| Published unsupervised BERT-base model | `218baf6bc48c463a53f5b4e1c568299537fab124` |
| BERT-base initialization model | `86b5e0934494bd15c9632b12f734a8a67f723594` |
| Princeton NLP SimCSE data | `e145e8bb659b2aa2669f32ef79cb4cdef6c58fef` |
| Manual Jittor reference, for later comparison only | `75f878a50ed9f5eb99a9b99885e95384abc24cf4` |

The source and reference checkouts are outside this repository in
`D:\科研\SimCSE-Benchmark-v1`. The reference checkout must stay outside both
candidate workspaces. Its model and loss differ from the upstream training code.
Its reported score does not establish numerical parity for this task.

The candidate seed is `D:\科研\SimCSE-Benchmark-v1\seed\SimCSE` at
`b74039ce07b3f289a8c177ff4999c2a2d0162294`. The `none` and `full` checkouts
were cloned from that commit and were clean at creation. The seed has a fixed
inference entry and an unimplemented `run.py`. The current [prompt](prompt.md)
is for the inference pilot. The training entry and prompt are not frozen yet.

The asset manifest now lists 11 downloaded files, totaling 1,088,800,072
bytes. All 11 SHA-256 hashes matched the pinned public metadata on 2026-10-27.
The published SimCSE model has 200 state tensors and 109,482,752 parameters.
Its strict FP32 NPZ archive has SHA-256
`0b1a8d07d607d6c23660ea9b80ae729ab37ff30ee93f35db2c7e8a3afbe8a4bb`.
The source adapter gave bitwise identical embeddings to `AutoModel` for four
fixed sentences on the remote PyTorch environment. The raw result is in
`D:\科研\SimCSE-Benchmark-v1\controller\prep\source-validation.json`.
Four inference TaskSpecs (smoke and full STS-B for each condition) are in
`D:\科研\SimCSE-Benchmark-v1\controller\tasks-cache-root-nomkl`. They use the
same source commit, weight archive, inputs, and seed. Each task disables
Jittor's optional MKL and CUTT setup. The target compilation cache stays on
the overlay filesystem. The data filesystem rejects new small files after
about 1,057 files in the tested directory tree.

The controller-only seed smoke check ended with the expected
`NotImplementedError` from `run.py`. The full STS-B seed check also ended with
the expected failure. Its source oracle ran twice and produced bitwise
identical 2,758-by-768 embeddings. The full check and raw report are in
`D:\科研\SimCSE-Benchmark-v1\controller\prep\simcse-seed-stsb-r2`. A previous
full check, `r1`, stopped during bundle extraction because the data filesystem
reached its small-file quota; it did not run the model. The remote execution
root now sits on the overlay filesystem, with weight and evidence symlinks to
the data filesystem. The first smoke bundle is archived locally in
`controller\prep\simcse-seed-smoke-r0\remote-bundle.tar.gz`.

Run `prepare_assets.py` to download the two pinned checkpoints, Wiki1M text,
and SentEval archive into `D:\科研\SimCSE-Benchmark-v1\assets`. The script saves
the public repository metadata and a SHA-256 manifest. Do not copy these
materials into a candidate workspace.

## Proposed pilot task

Use the published unsupervised SimCSE checkpoint for frozen inference. The
primary pilot score is Spearman correlation on the official STS-B test pairs.
Parse labels and sentences with the upstream SentEval format. Freeze tokenizer
files, a 64-token limit, pair order, and the scorer before the Agent trial.
The official test file has 1,379 pairs. Its longest tokenized sentence has 46
tokens, so the 64-token limit does not truncate any sentence. The frozen NPY
files and hashes are in `D:\科研\SimCSE-Benchmark-v1\assets\stsb-test-l64`.
Use `cls_before_pooler`. The upstream README assigns this pooler to
unsupervised SimCSE checkpoints. The upstream seven-task STS mean remains a
separate reference check; do not equate it with this pilot score.

The source oracle's STS-B test Spearman score is `0.7685349386294106`.
Upstream `evaluation.py` reports `76.85` for this one test, which agrees after
rounding. Its raw log is in
`D:\科研\SimCSE-Benchmark-v1\controller\prep\official-stsb-evaluation-r1.txt`.
The pilot tolerance in [tau.json](tau.json) is `0.005` Spearman. The source
repeat gap is zero, so this manual floor determines the tolerance. This is an
exploratory tolerance, not a validated formal acceptance threshold. The
target score and task metric remain undefined until an Agent candidate runs.

Use the BERT-base checkpoint and fixed Wiki1M text rows for a separate
training check. Freeze tokenized batches, initialization, random seeds, and
three to five optimizer steps. Disable dropout for the numerical training GATE.
Keep the inference and training weights separate.

Before a formal benchmark trial, finish these checks:

1. Verify the downloaded files against their recorded sizes and hashes.
2. Run the source model on the V100. Save source outputs and raw STS scores.
3. Prove that the source adapter reproduces the upstream model output and
   training step. The inference output check is done; the training check is not.
4. Build the same clean seed commit for `none` and `full`. Give both groups the
   same prompt, assets, environment, time limit, and query limit.
5. Freeze the formal task score tolerance, layer checkpoints, training GATE,
   and cost definitions. Preserve original JSON, JSONL, and NPZ evidence.

The V100 source Python is 3.10 with PyTorch 2.4.1. The upstream requirement
`transformers==4.2.1` could not install there because its old `tokenizers`
dependency had to build from source. A separate source package directory now
uses `transformers==4.26.1`; the upstream `BertForCL` imports under that
version. This is a pilot environment choice. Compare its published-checkpoint
output and STS score with the upstream behavior before freezing a formal task.

The V100 has 16 GiB of device memory. On 2026-10-26, the overlay filesystem
had no free blocks. The CLIP full-r3 trace directories were copied to the
attached data filesystem, checked with `rsync --checksum --dry-run`, and linked
back at their original paths. No trace evidence was discarded. The overlay
then had 1.4 GiB free, and the data filesystem had about 15 GiB free. Keep
large SimCSE assets and outputs on the data filesystem. A write probe then
reached `ENOSPC` after about 1,057 new small files, across two directories.
Its reported free inode count did not show this limit. Install Python packages
and keep the Jittor compilation cache on the overlay filesystem. The overlay
had about 1.2 GiB free after the source packages were installed. Confirm that
this is enough before an Agent trial.

Use `prepare_trial.py` to create one pilot controller. Set `TP_EXTRA_ROOTS` to
`D:\科研\SimCSE-Benchmark-v1` in the controller process. Before starting the
Agent, give each remote task path an overlay directory whose `evidence`
symlink points to the data filesystem. The task path digest includes the
controller path. Pre-stage the smoke, hidden STS-B, and official-delivery task
paths. The controller exposes only the smoke task to the Agent. It uses the
full STS-B task for the final hidden check. Keep the same prompt, budget,
container image, and proxy for the two conditions.

This repository is a possible NLP pilot. It is not yet part of the frozen
11-repository Benchmark v1 matrix. Do not label its pilot results as formal
benchmark results.
