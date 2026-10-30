"""Diagnostic B_l probe: record the 13 BERT checkpoint outputs on the frozen smoke input.

Runs one side per process. Checkpoints are `model.embeddings` and
`model.encoder.layer.{0..11}`; outputs are saved uncompressed in fp32 (fp64 for the
source reference path). This probe does not judge or sign any acceptance result.

Usage:
  python probe_layers.py --side source --repo SRC --weights W.npz --inputs DIR --output OUT.npz [--fp64]
  python probe_layers.py --side target --repo TGT --weights W.npz --inputs DIR --output OUT.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NAMES = ("input_ids", "attention_mask", "token_type_ids")
CHECKPOINTS = ["model.embeddings", *(f"model.encoder.layer.{i}" for i in range(12))]


def first_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def run_source(args, inputs, weights):
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    from torchport_adapter import TorchPortSimCSE

    model = TorchPortSimCSE()
    state = {k: torch.from_numpy(v) for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.cuda().eval()
    if args.fp64:
        model = model.double()
    modules = dict(model.named_modules())
    values, calls = {}, {}

    def hook(name):
        def record(_module, _inputs, output):
            calls[name] = calls.get(name, 0) + 1
            values[name] = first_tensor(output).detach().double().cpu().numpy()
        return record

    for name in CHECKPOINTS:
        modules[name].register_forward_hook(hook(name))
    tensors = [torch.from_numpy(inputs[n]).cuda() for n in NAMES]
    with torch.no_grad():
        output = model(*tensors)
    values["output"] = output.detach().double().cpu().numpy()
    return values, calls, {"missing": list(missing), "unexpected": list(unexpected)}


def run_target(args, inputs, weights):
    import jittor as jt

    jt.flags.use_cuda = 1
    from evaluation_entry import TorchPortSimCSE

    model = TorchPortSimCSE()
    own = set(model.state_dict())
    model.load_parameters({k: v for k, v in weights.items() if k in own})
    model.eval()
    modules = dict(model.named_modules())
    values, calls = {}, {}

    def hook(name):
        def record(_module, _inputs, output):
            calls[name] = calls.get(name, 0) + 1
            values[name] = np.asarray(first_tensor(output).numpy(), dtype=np.float64)
        return record

    for name in CHECKPOINTS:
        modules[name].register_forward_hook(hook(name))
    tensors = [jt.array(inputs[n]) for n in NAMES]
    with jt.no_grad():
        output = model(*tensors)
    values["output"] = np.asarray(output.numpy(), dtype=np.float64)
    load = {"missing": sorted(set(weights) - own), "unexpected": sorted(own - set(weights))}
    return values, calls, load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("source", "target"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp64", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.repo.resolve()))
    inputs = {n: np.load(args.inputs / f"smoke_{n}.npy", allow_pickle=False) for n in NAMES}
    with np.load(args.weights, allow_pickle=False) as archive:
        weights = {k: np.asarray(archive[k]) for k in archive.files}
    runner = run_source if args.side == "source" else run_target
    values, calls, load = runner(args, inputs, weights)
    missing = [n for n in CHECKPOINTS if n not in values]
    repeated = {n: c for n, c in calls.items() if c != 1}
    if missing or repeated:
        raise RuntimeError(json.dumps({"missing_checkpoints": missing, "repeated_calls": repeated}))
    dtype = np.float64 if args.fp64 else np.float32
    np.savez(args.output, **{k: v.astype(dtype) for k, v in values.items()})
    print(json.dumps({"side": args.side, "fp64": args.fp64, "checkpoints": len(CHECKPOINTS),
                      "shapes": {k: list(v.shape) for k, v in values.items()},
                      "load": load}))


if __name__ == "__main__":
    main()
