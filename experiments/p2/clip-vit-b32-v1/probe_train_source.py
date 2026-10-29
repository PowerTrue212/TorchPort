"""Run the fixed CLIP training pilot on the source GPU environment.

Run from a copy of the official CLIP source with torchport_adapter.py and
train_source_adapter.py on PYTHONPATH. Writes only a small JSON summary;
the frozen input files and checkpoint retain the raw reproducible materials.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from train_source_adapter import build_optimizer, load_train_model, train_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(20261028)
    torch.cuda.manual_seed_all(20261028)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = load_train_model()
    with np.load(args.weights, allow_pickle=False) as weights:
        status = model.model.load_state_dict(
            {name: torch.from_numpy(weights[name].copy()) for name in weights.files},
            strict=True,
        )
        loaded = len(weights.files)
    if status.missing_keys or status.unexpected_keys:
        raise RuntimeError("CLIP checkpoint did not load completely")
    model.cuda().train()
    state = build_optimizer(model)
    images = np.load(args.inputs / "images.npy", mmap_mode="r")
    tokens = np.load(args.inputs / "tokens.npy", mmap_mode="r")
    if images.shape != (4, 10, 3, 224, 224) or tokens.shape != (4, 10, 77):
        raise ValueError("CLIP training pilot input shape changed")

    losses: list[float] = []
    learning_rates: list[float] = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(images.shape[0]):
        learning_rates.append(float(state[0].param_groups[0]["lr"]))
        image_batch = torch.from_numpy(np.array(images[step], copy=True)).cuda()
        token_batch = torch.from_numpy(np.array(tokens[step], copy=True)).cuda()
        loss = train_step(model, state, image_batch, token_batch)
        losses.append(float(loss.detach().cpu()))
        if not np.isfinite(losses[-1]):
            raise RuntimeError(f"Non-finite CLIP loss at step {step}")
    result = {
        "schema_version": 1,
        "kind": "clip-train-source-feasibility-pilot",
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "weights_loaded": loaded,
        "steps": len(losses),
        "batch_size": 10,
        "losses": losses,
        "learning_rates": learning_rates,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "wall_seconds": time.monotonic() - started,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
