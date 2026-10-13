"""Migration submission contract.

Replace the two function bodies with a Jittor-native ViT-B/32 implementation.
The evaluator imports this file in the isolated Jittor environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_model(weights_path: Path) -> Any:
    """Build a jittor.Module and load weights from the shared FP32 NPZ archive."""
    raise NotImplementedError("migration agent must implement load_model")


def forward_logits(model: Any, images: np.ndarray, tokens: np.ndarray) -> Any:
    """Return one jittor.Var with shape [4, 4] (image-to-text CLIP logits)."""
    raise NotImplementedError("migration agent must implement forward_logits")
