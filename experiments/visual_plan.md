# Visualization Plan

This document defines model-specific visual evidence for TorchPort experiments.
The goal is not to draw generic activation maps for every model, but to reproduce
the visual evidence style that makes sense for each model family and compare:

```text
original PyTorch output
vs
migrated Jittor output
```

Each paired panel places `PyTorch` / `Jittor` above the image and the visual
type below it. Difference images are omitted; numerical differences are kept in
the accompanying table and line chart.

Metric names:

```text
RN = runnable
NA = numerical alignment
SA = strict alignment
```

## Model-specific focus

| Case | Model family | Paper-style visual evidence | TorchPort alignment visual |
|---|---|---|---|
| `vit_pytorch` | ViT | class-token attention / attention rollout, patch-token features | paired class-token attention and patch feature maps |
| `mae` | MAE | masked image, visible patches, reconstructed image, reconstruction error | paired reconstruction and decoder attention maps; mask map once `mask/ids_restore` are captured |
| `swin` | Swin Transformer | hierarchical stage feature maps, window/shifted-window attention, multi-scale backbone features | paired window attention and late-stage feature maps |
| `resnet_cifar10` | ResNet | residual block response and feature-map evolution by depth | paired early-block and late-block feature maps |
| `senet` | SE-ResNet | channel excitation weights and channel recalibration | paired channel-weight curves and SE block outputs |
| `pytorch_unet` | UNet | segmentation map, encoder/decoder feature maps, skip-connection features | paired encoder feature, decoder feature, and output-logit maps |
| `superres` | ESPCN-like super resolution | super-resolved output and high-frequency detail | paired SR output and high-frequency maps |
| `vae` | VAE | reconstruction and latent distribution | paired reconstruction maps and latent mean/standard-deviation curves |

## Priority figures

### Figure A — ViT attention and feature agreement

Panels: paired class-token attention and patch feature maps.

Current status:

```text
attention and feature tensors are already captured.
current diff is approximately 1e-8 to 1e-7.
```

### Figure B — MAE reconstruction agreement

Panels: paired reconstruction images and final decoder-block attention maps.

Current status:

```text
decoder_pred is captured and reconstruction can be unpatchified.
mask / ids_restore are not currently captured, so true masked-image visualization
requires one additional capture rule.
```

### Figure C — model family scorecard

Rows:

```text
ViT
MAE
Swin
ResNet
SENet
UNet
SuperRes
VAE
```

Columns:

```text
RN
NA
SA
main visualization available
paper-style visualization available
```

### Figure D — visual alignment curve

The line chart shows the maximum absolute difference of each model's displayed
representations on a log scale.

Purpose:

```text
Provide a compact numerical companion to the paired visuals without adding a
third `PT-JT` image column.
```

## Implementation notes

The visualizations must be generated from captured tensors or re-run certificates.
Do not synthesize feature maps.

For each model, save:

```text
experiments/data/tensors/<case>.npz
experiments/figures/fig_<case>_<topic>/plot.py
experiments/figures/fig_<case>_<topic>/plot.png
experiments/figures/fig_<case>_<topic>/figure-spec.md
experiments/figures/fig_<case>_<topic>/audit.md
```

The main paper/README should show model-specific visualizations, not only RN/NA/SA
tables. RN/NA/SA should be a compact summary, while visual evidence should carry
the claim that the migrated model behaves like the original.
