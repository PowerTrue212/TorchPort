"""Source-side CLIP training entry for a four-step feasibility pilot.

The paired batches contain one image and one text prompt for each CIFAR-10
class. This entry is not yet a Benchmark v1 training GATE or an Agent task.
"""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as functional
from torchport_adapter import TorchPortCLIP

STEPS = 4


def _layer_norm_for_calibration(module, value):
    """Keep upstream FP32 behavior and allow a true FP64 reference path.

    Upstream CLIP always casts LayerNorm inputs to FP32 for fp16 support. After
    the Harness converts the whole source model to FP64, that cast conflicts
    with FP64 weights. This branch applies only to the FP64 reference model.
    """
    if value.dtype == torch.float64:
        return functional.layer_norm(
            value, module.normalized_shape, module.weight, module.bias, module.eps
        )
    original_dtype = value.dtype
    result = functional.layer_norm(
        value.float(), module.normalized_shape, module.weight, module.bias, module.eps
    )
    return result.to(original_dtype)


def load_train_model() -> TorchPortCLIP:
    model = TorchPortCLIP()
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm) and type(module).__name__ == "LayerNorm":
            module.forward = MethodType(_layer_norm_for_calibration, module)
    return model


def build_optimizer(model: TorchPortCLIP):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, (STEPS - step) / STEPS)
    )
    return optimizer, scheduler


def train_step(model: TorchPortCLIP, state, images, tokens):
    optimizer, scheduler = state
    optimizer.zero_grad(set_to_none=True)
    logits = model(images, tokens)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = (functional.cross_entropy(logits, labels)
            + functional.cross_entropy(logits.t(), labels)) / 2
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss
