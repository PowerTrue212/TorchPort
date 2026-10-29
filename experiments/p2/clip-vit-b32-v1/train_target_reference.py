"""Controller-only Jittor CLIP training calibration candidate.

This starts from the full-r3 inference candidate, which passed the frozen
inference check. Its CLIP.execute uses no_grad, so this entry computes the same
logits through the model's components with gradients enabled. It is a reference
candidate to test, not a claimed correct target implementation.
"""

from __future__ import annotations

import jittor as jt
from jittor import nn
from run import load_model

STEPS = 4


class TorchPortCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = load_model()

    def execute(self, images, tokens):
        clip = self.model
        clip._ensure_fp32()
        image_features = clip.encode_image(images)
        text_features = clip.encode_text(tokens)
        image_features = image_features / jt.sqrt(
            (image_features * image_features).sum(dim=1, keepdims=1)
        )
        text_features = text_features / jt.sqrt(
            (text_features * text_features).sum(dim=1, keepdims=1)
        )
        return jt.exp(clip.logit_scale) * jt.matmul(
            image_features, text_features.transpose(0, 1)
        )


def load_train_model():
    return TorchPortCLIP()


def build_optimizer(model):
    # The inference loader calls eval(), which stops gradients in Jittor.
    # Re-enable them after the Harness has loaded the initial checkpoint.
    model.train()
    parameters = model.parameters()
    for parameter in parameters:
        parameter.start_grad()
    optimizer = jt.optim.AdamW(
        parameters, lr=1e-5, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0,
    )
    optimizer.param_groups[0]["lr"] = 1e-5
    optimizer._clip_step = 0
    return optimizer


def train_step(model, optimizer, images, tokens):
    logits = model(images, tokens)
    labels = jt.arange(logits.shape[0]).int32()
    loss = (nn.cross_entropy_loss(logits, labels)
            + nn.cross_entropy_loss(logits.transpose(0, 1), labels)) / 2
    optimizer.step(loss)
    step = optimizer._clip_step + 1
    optimizer._clip_step = step
    next_lr = 1e-5 * max(0.0, (STEPS - step) / STEPS)
    optimizer.lr = next_lr
    optimizer.param_groups[0]["lr"] = next_lr
    return loss
