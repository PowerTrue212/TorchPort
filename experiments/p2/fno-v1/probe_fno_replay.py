"""Replay the trained Darcy checkpoint on the 50 test samples and record three checkpoints.

Checkpoints: `lifting` output, `fno_blocks` output after the last layer, `projection`
output (= prediction). One side per process. Source runs the frozen controller adapter
(FP32, or FP64 with `--fp64`); target runs a frozen candidate's `load_train_model()`.
Diagnostic only; it does not judge or sign any acceptance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

CHECKPOINTS = ("lifting", "fno_blocks", "projection")
BATCH = 16


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_numpy(value) -> np.ndarray:
    value = value[0] if isinstance(value, (tuple, list)) else value
    if type(value).__module__.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def run_source(args, weights, x):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch  # noqa: PLC0415

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(args.repo))
    from torchport_train_adapter import load_train_model  # noqa: PLC0415

    model = load_train_model()
    model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()}, strict=True)
    model = model.cuda().eval()
    if args.fp64:
        model = model.double()
    last: dict[str, np.ndarray] = {}
    for name in CHECKPOINTS:
        getattr(model, name).register_forward_hook(
            lambda _m, _i, out, name=name: last.__setitem__(name, as_numpy(out)))
    dtype = torch.float64 if args.fp64 else torch.float32
    collected = {name: [] for name in (*CHECKPOINTS, "output")}
    with torch.no_grad():
        for start in range(0, len(x), BATCH):
            last.clear()
            out = model(torch.from_numpy(x[start:start + BATCH]).to("cuda", dtype))
            for name in CHECKPOINTS:
                collected[name].append(last[name])
            collected["output"].append(as_numpy(out))
    return collected, {"loaded": len(weights)}


def run_target(args, weights, x):
    import jittor as jt  # noqa: PLC0415

    jt.flags.use_cuda = 1
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    import evaluation_entry as entry  # noqa: PLC0415

    model = entry.load_train_model()
    model.load_parameters(weights)
    state = model.state_dict()
    mismatched = [k for k, v in weights.items()
                  if k not in state or not np.array_equal(as_numpy(state[k]), v)]
    if mismatched:
        raise ValueError(f"checkpoint parameters failed round-trip: {mismatched}")
    last: dict[str, np.ndarray] = {}
    for name in CHECKPOINTS:
        getattr(model, name).register_forward_hook(
            lambda _m, _i, out, name=name: last.__setitem__(name, as_numpy(out)))
    collected = {name: [] for name in (*CHECKPOINTS, "output")}
    with jt.no_grad():
        for start in range(0, len(x), BATCH):
            last.clear()
            out = model(jt.array(x[start:start + BATCH]))
            for name in CHECKPOINTS:
                collected[name].append(last[name])
            collected["output"].append(as_numpy(out))
    return collected, {"loaded": len(weights), "round_trip_mismatch": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("source", "target"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True, help="task-metrics-r2 directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp64", action="store_true")
    args = parser.parse_args()
    # target 侧会 chdir 到候选目录；输出路径必须先转成绝对路径
    args.repo, args.metrics = args.repo.resolve(), args.metrics.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.metrics / "weights.npz", allow_pickle=False) as archive:
        weights = {k: np.asarray(archive[k]) for k in archive.files}
    x = np.load(args.metrics / "test-inputs.npy", allow_pickle=False)
    with np.load(args.metrics / "test-predictions.npz", allow_pickle=False) as archive:
        target = np.asarray(archive["target"], dtype=np.float64)
    runner = run_source if args.side == "source" else run_target
    collected, load = runner(args, weights, x)
    values = {name: np.concatenate(parts) for name, parts in collected.items()}
    pred = values["output"].astype(np.float64)
    flat = (len(pred), -1)
    relative = (np.linalg.norm((pred - target).reshape(flat), axis=1)
                / np.linalg.norm(target.reshape(flat), axis=1))
    dtype = np.float64 if args.fp64 else np.float32
    np.savez(args.output, relative_l2=relative,
             **{k: v.astype(dtype) for k, v in values.items()})
    print(json.dumps({"side": args.side, "fp64": args.fp64, "repo": args.repo.name, **load,
                      "weights_sha256": sha256(args.metrics / "weights.npz"),
                      "shapes": {k: list(v.shape) for k, v in values.items()},
                      "mean_relative_l2": float(relative.mean())}))


if __name__ == "__main__":
    main()
