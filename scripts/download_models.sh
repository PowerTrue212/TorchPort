#!/usr/bin/env bash
set -euo pipefail

HF_CLI="${HF_CLI:-hf}"
MODEL_ROOT="${MODEL_ROOT:-artifacts/models}"

mkdir -p "${MODEL_ROOT}/cnn-resnet18" "${MODEL_ROOT}/vit-tiny-patch16"

"${HF_CLI}" download \
  timm/resnet18.a1_in1k \
  config.json model.safetensors \
  --revision 491b427b45c94c7fb0e78b5474cc919aff584bbf \
  --local-dir "${MODEL_ROOT}/cnn-resnet18"

"${HF_CLI}" download \
  timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k \
  config.json model.safetensors \
  --revision 20ea349d1a6fc5fa69a3ae25add5ffae05f228de \
  --local-dir "${MODEL_ROOT}/vit-tiny-patch16"

