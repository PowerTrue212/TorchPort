"""Agent-owned CamoFormer migration contract.

Implement ``load_model`` and return a Jittor module whose call accepts one NCHW
float32 image and returns the five CamoFormer prediction tensors in P5..P1 order.
TorchPort loads the frozen PyTorch state into the returned module before inference.
"""

from __future__ import annotations

import jittor as jt


def load_model() -> jt.Module:
    raise NotImplementedError("Implement the Jittor CamoFormer model")
