"""Stable TorchPort entry around the Agent-owned CLIP submission contract."""

from __future__ import annotations

from pathlib import Path

import jittor as jt
from submission import forward_logits, load_model


class TorchPortCLIP(jt.Module):
    def __init__(self, weights_path: str) -> None:
        self.model = load_model(Path(weights_path))
        self.model.visual.transformer.resblocks = jt.nn.ModuleList(
            self.model.visual.transformer.resblocks
        )
        self.model.transformer.resblocks = jt.nn.ModuleList(self.model.transformer.resblocks)

    def execute(self, images: jt.Var, tokens: jt.Var) -> jt.Var:
        return forward_logits(self.model, images.numpy(), tokens.numpy())
