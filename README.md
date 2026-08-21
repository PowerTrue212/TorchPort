<div align="center">
  <img src="docs/assets/torchport-showcase.png" alt="TorchPort evidence-guided migration" width="100%">

  <h1>TorchPort</h1>

  <p>
    <strong>DeepSeek-powered, evidence-guided, fully domestic AI migration platform.</strong><br>
    Convert PyTorch research repositories to Jittor with verified execution, numerical evidence,
    repair loops, and native readiness for Ascend-class domestic accelerators.
  </p>

  <p>
    <img alt="python" src="https://img.shields.io/badge/python-%E2%89%A53.10-3776AB?logo=python&logoColor=white">
    <img alt="DeepSeek" src="https://img.shields.io/badge/LLM-DeepSeek-4B6CFF">
    <img alt="Jittor" src="https://img.shields.io/badge/target-Jittor-2F6BFF">
    <img alt="Ascend" src="https://img.shields.io/badge/accelerator-Ascend-E6242E">
    <img alt="DSH" src="https://img.shields.io/badge/agent-DSH-7A5AF8">
    <img alt="evidence" src="https://img.shields.io/badge/method-evidence--guided-F27CA5">
  </p>

  <p>
    <a href="#quick-start">Quick Start</a> ·
    <a href="#workflow">Workflow</a> ·
    <a href="#dsh-integration">DSH</a> ·
    <a href="#figures">Figures</a> ·
    <a href="#status">Status</a> ·
    <a href="README_zh.md">中文 README</a>
  </p>
</div>

## Architecture

<div align="center">
  <img src="docs/assets/torchport-architecture.png" alt="TorchPort architecture" width="100%">
</div>

TorchPort is designed as a fully domestic conversion stack: DeepSeek provides repository-level
reasoning, DSH orchestrates tool calls and repair loops, Jittor is the domestic target framework, and
Ascend-class accelerators are a natural deployment target. The platform is built around numerical
evidence rather than blind source rewriting.

## Overview

TorchPort is a DeepSeek/DSH-based migration platform for domestic AI infrastructure. Its first
target path is PyTorch to Jittor, with the system architecture designed for domestic compute backends
such as Ascend. The project goal is not just source-code conversion. TorchPort treats migration as a
verified loop:

```text
Unknown PyTorch repo
        -> DeepSeek/DSH analyzes the repository
        -> TorchPort migrates and runs PyTorch/Jittor
        -> tensor traces are compared
        -> the first divergence is localized
        -> DSH repairs with evidence
        -> a verified Jittor repo is produced
```

TorchPort separates responsibilities clearly:

- DeepSeek/DSH makes repository-specific decisions.
- TorchPort provides deterministic tools and evidence.
- Jittor is the domestic target framework.
- Ascend-class accelerator support is a first-class direction in the platform design.

## Why TorchPort

Most framework converters answer: "Can the code be rewritten?"

TorchPort asks a stricter question: "Does the migrated project really run, load the same weights,
receive the same inputs, and preserve numerical behavior?"

The core signals are:

- `RN`: migrated Jittor project runs.
- `NA`: final PyTorch/Jittor outputs are numerically aligned.
- `SA`: stricter intermediate-layer alignment passes.
- `First divergence`: the first layer or operator where a meaningful mismatch appears.
- `Certificate`: a reproducible report with inputs, weights, traces, errors, and failure attribution.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run checks:

```bash
pytest tests -q
ruff check .
python -m build
```

Scan and plan a PyTorch repository:

```bash
PYTHONPATH=src python -m torchport scan /path/to/pytorch-repo
PYTHONPATH=src python -m torchport plan /path/to/pytorch-repo --target jittor-native
```

Run an automatic migration attempt:

```bash
PYTHONPATH=src python -m torchport auto /path/to/pytorch-repo \
  --output /path/to/migrated-jittor \
  --target jittor-native \
  --entrypoint main.py \
  --model main.Model \
  --shape 1,3,224,224 \
  --check-run
```

The main report is written to:

```text
/path/to/migrated-jittor/.torchport/auto.json
```

## Workflow

```text
Repository scan
      -> entry recovery
      -> migration
      -> source/target execution
      -> weight and input checks
      -> tensor trace comparison
      -> first-divergence localization
      -> repair proposal
      -> re-verification
      -> certificate
```

This loop is what makes TorchPort different from a static conversion script. The model-specific
knowledge is not hard-coded as adapter files. DSH reads the repository and uses TorchPort evidence to
decide the next action.

## DSH Integration

The DSH plugin lives in `integrations/dsh` and exposes eight structured tools:

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

Run plugin tests:

```bash
cd integrations/dsh
npm install
npm test
```

## Status

Implemented:

- repository scanning and model-entry recovery;
- PyTorch-to-native-Jittor migration skeleton;
- controlled PyTorch/Jittor execution;
- weight transfer checks;
- tensor trace collection;
- FP32 and strict numerical comparison;
- first-divergence localization;
- structured failure codes;
- DSH plugin with repair/verify loop;
- experiment scripts and publication-style figures.

Evidence boundary:

- The formal DSH end-to-end run is PyTorch examples SuperRes.
- Historical certificates and visualizations cover multiple model families.
- Full multi-repository comparison against official converter, JTorch, pure Codex, and pure DSH is
  still a benchmark-stage task.
- Ascend-class support is a platform direction and interface target; the current public evidence is
  centered on PyTorch/Jittor migration and numerical verification.

## Figures

### Main Experimental Figures

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

### Model Visualizations

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

## SuperRes DSH Result

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

The migrated Jittor project runs and final FP32 output alignment passes. The stricter intermediate
check reports a trace-level divergence at `conv3`; TorchPort keeps this in the certificate instead of
hiding it.

## Layout

```text
src/torchport/       Python core and CLI
integrations/dsh/    DSH plugin
benchmarks/          entry-pattern and repository benchmarks
compatibility/       compatibility notes and repair rules
experiments/         experiment data, figures, prompts, and reports
docs/                design notes and assets
tests/               unit and integration tests
```

## Reproduce Figures

```bash
python experiments/scripts/build.py
python experiments/figures/fig_e2e/plot.py
python experiments/figures/fig_rounds/plot.py
```

Method-figure prompt:

```text
experiments/prompts/architecture.md
```

## License

Apache-2.0. See `LICENSE`.
