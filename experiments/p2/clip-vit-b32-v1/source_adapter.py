"""Pilot source entry. The model implementation is the frozen upstream model.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _upstream_model_class():
    path = Path(__file__).resolve().parent / "clip" / "model.py"
    spec = importlib.util.spec_from_file_location("clip_upstream_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CLIP


class TorchPortCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        upstream = _upstream_model_class()
        self.model = upstream(
            embed_dim=512,
            image_resolution=224,
            vision_layers=12,
            vision_width=768,
            vision_patch_size=32,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
        )

    def forward(self, images, tokens):
        image_logits, _ = self.model(images, tokens)
        return image_logits
