"""Stable TorchPort entry around the Agent-owned RAFT submission contract."""

from __future__ import annotations

from pathlib import Path

import jittor as jt
from submission import load_model


class TorchPortRAFT(jt.Module):
    def __init__(self, weights_path: str) -> None:
        self.model = load_model(Path(weights_path))

    def execute(
        self, image1: jt.Var, image2: jt.Var
    ) -> tuple[jt.Var, jt.Var]:
        return self.model(image1, image2, iters=12, test_mode=True)
