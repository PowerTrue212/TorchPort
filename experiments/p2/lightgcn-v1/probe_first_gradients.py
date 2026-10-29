"""Capture first-step embedding gradients from the frozen LightGCN pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--side", choices=("source", "target"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    task = json.loads(args.task.read_text("utf-8"))
    side = task[args.side]
    sys.path.insert(0, side["repo"])
    users, positive, negative = (
        np.load(spec["path"], allow_pickle=False)[0] for spec in task["inputs"]
    )
    with np.load(task["weights"], allow_pickle=False) as archive:
        weights = {name: np.array(archive[name], copy=True) for name in archive.files}
    if args.side == "source":
        os.environ["TP_LIGHTGCN_DEPS"] = side["env"]["TP_LIGHTGCN_DEPS"]
        import torch  # noqa: PLC0415
        import torchport_train_adapter as entry  # noqa: PLC0415

        torch.use_deterministic_algorithms(True)
        model = entry.load_train_model()
        model.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})
        model.train()
        tensors = [torch.from_numpy(value).cuda() for value in (users, positive, negative)]
        bpr, regularization = model.bpr_loss(*tensors)
        loss = bpr + 1e-4 * regularization
        loss.backward()
        torch.cuda.synchronize()
        gradients = {name: param.grad.detach().cpu().numpy()
                     for name, param in model.named_parameters()}
        loss_value = float(loss.detach().cpu())
    else:
        import jittor as jt  # noqa: PLC0415
        import evaluation_entry as entry  # noqa: PLC0415

        jt.flags.use_cuda = 1
        model = entry.load_train_model()
        model.load_parameters(weights)
        optimizer = entry.build_optimizer(model)
        tensors = [jt.array(value) for value in (users, positive, negative)]
        loss = model.bpr_loss(*tensors)
        optimizer.backward(loss)
        gradients = {name: np.array(param.opt_grad(optimizer).numpy(), copy=True)
                     for name, param in model.named_parameters()}
        loss_value = float(loss.numpy().reshape(-1)[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **gradients)
    report = {"side": args.side, "loss": loss_value,
              "gradients": {name: {"shape": list(value.shape),
                                    "l2": float(np.linalg.norm(value.astype(np.float64))),
                                    "max_abs": float(np.abs(value).max())}
                            for name, value in gradients.items()}}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
