# TorchPort

TorchPort is an evidence-guided PyTorch-to-Jittor migration system built for unknown research
repositories. It is not a collection of model-specific adapters. The intended workflow is:

```text
Unknown PyTorch repo
        -> DSH / LLM analyzes the repo
        -> TorchPort migrates and runs both sides
        -> numerical traces are compared
        -> the first divergence is localized
        -> DSH repairs with evidence
        -> the migrated Jittor repo is verified
```

TorchPort keeps the deterministic parts in Python and exposes them to DSH as structured tools. DSH
decides what to try next; TorchPort supplies facts: entry recovery, migration artifacts, runtime logs,
tensor traces, failure codes, patches, and certificates.

## Why

Most converters answer only one question: "can the code be rewritten?" TorchPort targets the harder
question: "does the migrated project really run and preserve numerical behavior?"

The core outputs are:

- `RN`: the migrated Jittor project runs.
- `NA`: final PyTorch/Jittor outputs are numerically aligned.
- `SA`: stricter intermediate-layer alignment passes.
- `First divergence`: the first layer/operator where a significant mismatch appears.
- `Certificate`: a reproducible JSON report containing inputs, weights, traces, errors, and failure
  attribution.

## Current Status

Implemented:

- repository scanning and model-entry recovery;
- PyTorch-to-native-Jittor migration skeleton;
- controlled source/target execution;
- weight transfer checks;
- tensor trace collection;
- FP32 and strict numerical comparison;
- first-divergence localization;
- structured failure codes;
- DSH plugin with a repair/verify loop;
- experiment scripts and publication-style figures.

Current evidence boundary:

- The formal DSH end-to-end run is PyTorch examples SuperRes.
- Historical model certificates and visualizations cover several model families.
- Multi-repository official/JTorch/Codex/pure-DSH baselines still need a full frozen benchmark run.

## DSH Integration

The DSH plugin is in `integrations/dsh`. It exposes eight tools:

```text
scan_repository
create_migration_plan
start_migration
run_verification
locate_first_divergence
propose_repairs
apply_verified_repair
generate_certification_report
```

The plugin does not hard-code ResNet, ViT, Swin, DiT, or other model adapters. It passes structured
TorchPort evidence to DSH, and DSH chooses repository-specific migration or repair actions.

```bash
cd integrations/dsh
npm install
npm test
```

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run local checks:

```bash
pytest tests -q
ruff check .
python -m build
```

## Basic Usage

Scan and plan a repository:

```bash
PYTHONPATH=src python -m torchport scan /path/to/pytorch-repo
PYTHONPATH=src python -m torchport plan /path/to/pytorch-repo --target jittor-native
```

Run an automatic migration and verification attempt:

```bash
PYTHONPATH=src python -m torchport auto /path/to/pytorch-repo \
  --output /path/to/migrated-jittor \
  --target jittor-native \
  --entrypoint main.py \
  --model main.Model \
  --shape 1,3,224,224 \
  --check-run
```

The main result is written to:

```text
/path/to/migrated-jittor/.torchport/auto.json
```

## Repository Layout

```text
src/torchport/       Python core and CLI
integrations/dsh/    DSH plugin
benchmarks/          entry-pattern and repository benchmarks
compatibility/       compatibility notes and repair rules
experiments/         experiment data, figures, prompts, and reports
docs/                design notes
tests/               unit and integration tests
```

## Main Experimental Figures

These figures are generated from files under `experiments/`. They are included here so the repository
front page shows both the method and the current evidence.

| End-to-end comparison | DSH repair trajectory |
| --- | --- |
| <img src="experiments/figures/fig_e2e/plot.png" width="430"> | <img src="experiments/figures/fig_rounds/plot.png" width="430"> |

| Method summary | Model coverage snapshot |
| --- | --- |
| <img src="experiments/figures/fig_methods/plot.png" width="430"> | <img src="experiments/figures/fig_models/plot.png" width="430"> |

| Tensor evidence | Failure/status view |
| --- | --- |
| <img src="experiments/figures/fig_tensors/plot.png" width="430"> | <img src="experiments/figures/fig_status/plot.png" width="430"> |

| Strict alignment | Failure attribution |
| --- | --- |
| <img src="experiments/figures/fig_strict/plot.png" width="430"> | <img src="experiments/figures/fig_fail/plot.png" width="430"> |

## Model Visualizations

The model visualizations compare PyTorch and Jittor internal evidence under matched inputs and
weights. They are visual evidence, not a replacement for full dataset-level accuracy evaluation.

| ViT attention/features | MAE reconstruction |
| --- | --- |
| <img src="experiments/figures/fig_vit/plot.png" width="430"> | <img src="experiments/figures/fig_mae/plot.png" width="430"> |

| ViT/MAE overview | ViT alignment |
| --- | --- |
| <img src="experiments/figures/fig_vit_mae/plot.png" width="430"> | <img src="experiments/figures/fig_vit_align/plot.png" width="430"> |

| MAE alignment | Swin attention/features |
| --- | --- |
| <img src="experiments/figures/fig_mae_align/plot.png" width="430"> | <img src="experiments/figures/fig_swin/plot.png" width="430"> |

| ResNet feature maps | SENet channel response |
| --- | --- |
| <img src="experiments/figures/fig_resnet/plot.png" width="430"> | <img src="experiments/figures/fig_senet/plot.png" width="430"> |

| U-Net segmentation evidence | Super-resolution output |
| --- | --- |
| <img src="experiments/figures/fig_unet/plot.png" width="430"> | <img src="experiments/figures/fig_superres/plot.png" width="430"> |

| VAE latent/reconstruction | Visual summary |
| --- | --- |
| <img src="experiments/figures/fig_vae/plot.png" width="430"> | <img src="experiments/figures/fig_visual/plot.png" width="430"> |

## DSH SuperRes Result

The current formal DSH-run sample is PyTorch examples SuperRes:

```text
Official converter:
  CV=true
  RN=false
  failure=jittor.init.orthogonal_ is missing

DSH + TorchPort:
  CV=true
  RN=true
  NA=true
  SA=false
  first strict divergence=conv3
```

This means the migrated Jittor project ran and final FP32 output alignment passed. The stricter
intermediate-layer check still reports a trace-level divergence at `conv3`, which is kept in the
certificate instead of being hidden.

## Reproduce Figures

```bash
python experiments/scripts/build.py
python experiments/figures/fig_e2e/plot.py
python experiments/figures/fig_rounds/plot.py
```

Architecture prompt for generating the method figure:

```text
experiments/prompts/architecture.md
```

## License

Apache-2.0. See `LICENSE`.
