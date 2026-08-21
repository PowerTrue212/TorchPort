<div align="center">
  <img src="docs/assets/torchport-showcase.png" alt="TorchPort migration and verification" width="100%">

  <h1>TorchPort</h1>

  <p>
    <strong>A PyTorch → Jittor migration and verification system for a domestic-first AI software stack.</strong><br>
    TorchPort uses DeepSeek/DSH for repository-level analysis and repair decisions, targets Jittor first,
    and is designed around Ascend-class domestic accelerator backends.
  </p>

  <p>
    <img alt="python" src="https://img.shields.io/badge/python-%E2%89%A53.10-3776AB?logo=python&logoColor=white">
    <img alt="DeepSeek" src="https://img.shields.io/badge/LLM-DeepSeek-4B6CFF">
    <img alt="Jittor" src="https://img.shields.io/badge/target-Jittor-2F6BFF">
    <img alt="Ascend" src="https://img.shields.io/badge/accelerator-Ascend-E6242E">
    <img alt="DSH" src="https://img.shields.io/badge/agent-DSH-7A5AF8">
    <img alt="evidence" src="https://img.shields.io/badge/method-verification-F27CA5">
  </p>

  <p>
    <a href="README.md">中文</a> ·
    English
  </p>

  <p>
    <a href="#architecture">Architecture</a> ·
    <a href="#positioning">Positioning</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#dsh-integration">DSH Integration</a> ·
    <a href="#figures">Figures</a> ·
    <a href="#status">Status</a>
  </p>
</div>

## Architecture

<div align="center">
  <img src="docs/assets/torchport-architecture.png" alt="TorchPort architecture" width="100%">
</div>

TorchPort starts from a narrow requirement: migration should be judged by execution and numerical behavior, not by whether source text has been rewritten.

The system has two layers. DeepSeek/DSH reads unknown PyTorch repositories and schedules migration or repair actions. TorchPort provides deterministic tools for scanning, migration, dual-runtime execution, numerical alignment, and first-divergence localization. Jittor is the primary target today. Ascend-class domestic accelerators are treated as backend constraints from the design stage, not as an afterthought. The long-term goal is a verifiable migration path from DeepSeek/DSH and Jittor to domestic compute backends.

## Positioning

TorchPort is not a collection of model-specific adapters, and it is not a one-pass conversion script. It addresses a reproducible migration question:

```text
Given an unknown PyTorch repository,
can the system migrate it to Jittor,
run both PyTorch and Jittor with the same input and weights,
compare numerical outputs and intermediate traces,
localize the first divergence when alignment fails,
and record the repair process?
```

This gives each component a clear role:

- DeepSeek/DSH handles repository-specific understanding, including entry points, configs, weights, and inference paths.
- TorchPort produces auditable engineering evidence, including scan results, execution logs, tensor traces, and numerical certificates.
- Jittor is the current domestic target framework with the most complete verification path.
- Ascend-class accelerators are a backend target for later integration under the same evidence chain.

## Domestic Stack Path

TorchPort treats the domestic conversion platform as a sequence of verifiable layers:

| Layer | Current choice | Role |
| --- | --- | --- |
| Agent interface | DeepSeek/DSH | Read unknown repositories and choose migration or repair actions |
| Target framework | Jittor | Receive migrated PyTorch code and weights |
| Verification | dual execution + tensor traces | Decide whether the migration result is trustworthy |
| Backend direction | Ascend-class domestic accelerators | Shape the deployment interface and future backend work |

The current public evidence focuses on PyTorch-to-Jittor migration and numerical verification. Full Ascend-class backend experiments are part of the next backend-integration stage.

## Why TorchPort

Most converters ask:

```text
Can the code be converted?
```

TorchPort asks:

```text
Does the migrated project run?
Did it load the same weights?
Did it receive the same input?
Are the final outputs numerically aligned?
If not, where does the first layer/operator-level divergence occur?
```

The README uses the following short metrics:

| Metric | Meaning |
| --- | --- |
| `RN` | The migrated Jittor project runs |
| `NA` | Final PyTorch/Jittor outputs are numerically aligned |
| `SA` | Intermediate traces pass a stricter alignment check |
| `FD` | First numerical divergence |
| `Cert` | A numerical certificate with inputs, weights, traces, errors, and failure attribution |

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run basic checks:

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

Run one automatic migration attempt:

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
repository scan
      -> model-entry recovery
      -> code migration
      -> PyTorch/Jittor dual execution
      -> weight and input checks
      -> tensor-trace comparison
      -> first-divergence localization
      -> repair proposal
      -> re-verification
      -> numerical certificate
```

TorchPort does not hard-code model knowledge into files such as `resnet_adapter.py` or `vit_adapter.py`. For a new repository, DSH reads the project and calls TorchPort tools. TorchPort returns structured evidence. DSH then chooses the next migration or repair action.

## DSH Integration

The DSH plugin lives in `integrations/dsh` and currently exposes eight tools:

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
- weight-transfer checks;
- tensor-trace collection;
- FP32 and strict numerical comparison;
- first-divergence localization;
- structured failure codes;
- DSH plugin with a repair/verification loop;
- experiment scripts and paper-style figures.

Evidence boundary:

- The formal DSH end-to-end sample is PyTorch examples SuperRes.
- Earlier dual-runtime certificates and visualizations cover several model families.
- A unified comparison against the official converter, JTorch, pure Codex, pure DSH, and DSH + TorchPort is still in the benchmark-freeze stage.
- Ascend-class domestic accelerators are part of the backend interface design, while the current public evidence is centered on PyTorch/Jittor migration and numerical verification.

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
  FD=conv3
```

This sample shows two things. First, the migrated Jittor project runs and its final FP32 output aligns. Second, the stricter intermediate check still records a trace-level divergence at `conv3`. TorchPort keeps that evidence in the certificate instead of hiding it.

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

Architecture prompt:

```text
experiments/prompts/architecture.md
```

## License

Apache-2.0. See `LICENSE`.
