"""Hash every first-step CLIP Jittor gradient before the optimizer update."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jittor as jt
import numpy as np
from evaluation_entry import build_optimizer, load_train_model
from jittor import nn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    model = load_train_model()
    with np.load(args.weights, allow_pickle=False) as weights:
        model.model.load_parameters({name: weights[name] for name in weights.files})
        loaded = len(weights.files)
    optimizer = build_optimizer(model)
    images = jt.array(np.array(np.load(args.images, mmap_mode="r")[0]))
    tokens = jt.array(np.array(np.load(args.tokens, mmap_mode="r")[0]))
    logits = model(images, tokens)
    labels = jt.arange(logits.shape[0]).int32()
    loss = (nn.cross_entropy_loss(logits, labels)
            + nn.cross_entropy_loss(logits.transpose(0, 1), labels)) / 2
    optimizer.backward(loss)

    names = {id(param): name for name, param in model.named_parameters()}
    records = []
    selected = {}
    for group in optimizer.param_groups:
        for parameter, gradient in zip(group["params"], group["grads"]):
            name = names[id(parameter)]
            array = gradient.numpy()
            records.append({
                "name": name,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "sum": float(np.sum(array.astype(np.float64))),
                "max_abs": float(np.max(np.abs(array))),
            })
            if name in {
                "model.visual.ln_pre.weight",
                "model.visual.transformer.resblocks.0.attn.in_proj_weight",
                "model.transformer.resblocks.4.attn.in_proj_weight",
                "model.logit_scale",
            }:
                selected[name] = array
    loss_value = float(loss.numpy().reshape(-1)[0])
    np.savez(args.selected, **selected)
    args.output.write_text(json.dumps({
        "loaded": loaded, "loss": loss_value,
        "gradient_count": len(records), "gradients": records,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"loss": loss_value, "gradient_count": len(records),
                      "selected": list(selected)}))


if __name__ == "__main__":
    main()
