"""Stable TorchPort entry around the Agent-owned CamoFormer submission."""

from __future__ import annotations

import jittor as jt
from submission import load_model


class TorchPortCamoFormer(jt.Module):
    def __init__(self) -> None:
        self.model = load_model()

    def execute(self, image: jt.Var) -> tuple[jt.Var, ...]:
        return self.model(image)
