"""Capture CLIP Jittor image/text features and loss before any update.

Run against the controller-only calibration candidate on the V100. This probe
does not decide parity; it narrows where repeated target runs first differ.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jittor as jt
import numpy as np
from evaluation_entry import load_train_model
from jittor import nn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-fuse", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    if args.no_fuse:
        jt.flags.no_fuse = 1
    model = load_train_model()
    with np.load(args.weights, allow_pickle=False) as weights:
        model.model.load_parameters({name: weights[name] for name in weights.files})
        loaded = len(weights.files)
    model.train()
    for parameter in model.parameters():
        parameter.start_grad()
    images = jt.array(np.array(np.load(args.images, mmap_mode="r")[0]))
    tokens = jt.array(np.array(np.load(args.tokens, mmap_mode="r")[0]))
    started = time.monotonic()
    clip = model.model
    image_features = clip.encode_image(images)
    text_features = clip.encode_text(tokens)
    image_features = image_features / jt.sqrt(
        (image_features * image_features).sum(dim=1, keepdims=1)
    )
    text_features = text_features / jt.sqrt(
        (text_features * text_features).sum(dim=1, keepdims=1)
    )
    logits = jt.exp(clip.logit_scale) * jt.matmul(
        image_features, text_features.transpose(0, 1)
    )
    labels = jt.arange(logits.shape[0]).int32()
    loss = (nn.cross_entropy_loss(logits, labels)
            + nn.cross_entropy_loss(logits.transpose(0, 1), labels)) / 2
    arrays = {"image_features": image_features.numpy(),
              "text_features": text_features.numpy(),
              "logits": logits.numpy(), "loss": loss.numpy()}
    np.savez(args.output, **arrays)
    print(json.dumps({"loaded": loaded, "no_fuse": args.no_fuse,
                      "wall_seconds": time.monotonic() - started,
                      "loss": float(arrays["loss"].reshape(-1)[0])}))


if __name__ == "__main__":
    main()
