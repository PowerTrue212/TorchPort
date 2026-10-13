"""Common RAFT migration submission contract.

Each method implements this interface inside its own workspace. The common
evaluator supplies the frozen checkpoint and the two prepared NumPy inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_model(weights_path: Path) -> Any:
    """Build a native Jittor RAFT and load the frozen raft-things weights."""
    raise NotImplementedError("migration agent must implement load_model")


def forward_flow(model: Any, image1: np.ndarray, image2: np.ndarray) -> tuple[Any, Any]:
    """Return (flow_low, flow_up) using iters=12 and test_mode=True."""
    raise NotImplementedError("migration agent must implement forward_flow")
