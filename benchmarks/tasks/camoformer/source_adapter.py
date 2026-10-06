"""Stable TorchPort entry for the frozen official CamoFormer source."""

from __future__ import annotations

import torch
from model.CamoFormer import CamoFormer


class TorchPortCamoFormer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = CamoFormer(cfg=None)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.model(image)
