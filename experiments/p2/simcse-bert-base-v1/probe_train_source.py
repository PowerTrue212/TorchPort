"""Run a small deterministic source-only SimCSE training check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BASE_SHA256 = "097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a"
SEED = 20261027
STEPS = 2
LR = 3e-5
SELECTED = (
    "bert.encoder.layer.0.attention.self.query.weight",
    "bert.encoder.layer.11.output.LayerNorm.weight",
    "mlp.dense.weight",
    "mlp.dense.bias",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(str(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dropout", type=float, choices=(0.0, 0.1), default=0.0)
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--save-selected-tensors", action="store_true")
    parser.add_argument("--save-initial-state", action="store_true")
    args = parser.parse_args()
    if file_sha256(args.base_dir / "pytorch_model.bin") != BASE_SHA256:
        raise ValueError("BERT-base checkpoint differs from the pinned public file")
    batch_manifest = json.loads((args.batch_dir / "manifest.json").read_text("utf-8"))
    for name, info in batch_manifest["files"].items():
        if file_sha256(args.batch_dir / f"{name}.npy") != info["sha256"]:
            raise ValueError(f"Frozen batch hash differs: {name}")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch  # noqa: PLC0415
    from transformers import BertConfig  # noqa: PLC0415
    from transformers import __version__ as transformers_version

    sys.path.insert(0, str(args.source_repo.resolve()))
    from simcse.models import BertForCL  # noqa: PLC0415

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    config = BertConfig.from_json_file(str(args.base_dir / "config.json"))
    config.hidden_dropout_prob = args.dropout
    config.attention_probs_dropout_prob = args.dropout
    model_args = SimpleNamespace(pooler_type="cls", temp=0.05, do_mlm=False,
                                 mlp_only_train=True, hard_negative_weight=0.0)
    model, loading = BertForCL.from_pretrained(
        str(args.base_dir), config=config, model_args=model_args,
        local_files_only=True, output_loading_info=True,
    )
    model.cuda().train()
    if args.precision == "fp64":
        model.double()
    inputs = {name: torch.from_numpy(np.load(args.batch_dir / f"{name}.npy",
                                             allow_pickle=False)).cuda()
              for name in ("input_ids", "attention_mask", "token_type_ids")}
    if any(value.shape != (8, 2, 32) for value in inputs.values()):
        raise ValueError("Frozen batch shape must be [8, 2, 32]")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.0, foreach=False)
    parameters = dict(model.named_parameters())
    for name in SELECTED:
        if name not in parameters:
            raise KeyError(name)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.save_initial_state:
        np.savez(
            args.output_dir / "train_initial.npz",
            **{name: value.detach().cpu().numpy() for name, value in model.state_dict().items()},
        )
    selected_arrays = {}
    if args.save_selected_tensors:
        selected_arrays = {
            f"initial.param.{name}": parameters[name].detach().cpu().numpy().copy()
            for name in SELECTED
        }
    losses = []
    logits = []
    steps = []
    initial_digest = state_sha256(model)
    for index in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        similarity = []
        hook = model.sim.register_forward_hook(
            lambda _module, _inputs, result: similarity.append(result.detach().cpu().numpy())
        )
        try:
            output = model(**inputs, return_dict=True)
        finally:
            hook.remove()
        if len(similarity) != 1 or similarity[0].shape != (8, 8):
            raise ValueError(f"Unexpected similarity output at step {index}")
        loss = output.loss
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at step {index}")
        losses.append(float(loss.detach().cpu()))
        logits.append(similarity[0])
        loss.backward()
        grad_squares = [param.grad.detach().float().square().sum()
                        for param in model.parameters() if param.grad is not None]
        grad_norm = float(torch.stack(grad_squares).sum().sqrt().cpu())
        if not np.isfinite(grad_norm) or grad_norm <= 0:
            raise ValueError(f"Missing or non-finite gradient at step {index}")
        before = {name: parameters[name].detach().clone() for name in SELECTED}
        selected_grads = {name: float(parameters[name].grad.detach().float().norm().cpu())
                          for name in SELECTED}
        if args.save_selected_tensors:
            selected_arrays.update({
                f"step_{index}.grad.{name}": parameters[name].grad.detach().cpu().numpy().copy()
                for name in SELECTED
            })
        optimizer.step()
        updates = {name: float((parameters[name].detach() - before[name]).float().norm().cpu())
                   for name in SELECTED}
        if args.save_selected_tensors:
            selected_arrays.update({
                f"step_{index}.param.{name}": parameters[name].detach().cpu().numpy().copy()
                for name in SELECTED
            })
        steps.append({"index": index, "loss": losses[-1], "grad_norm": grad_norm,
                      "selected_grad_norms": selected_grads,
                      "selected_update_norms": updates,
                      "state_sha256_after_step": state_sha256(model)})

    np.save(args.output_dir / "losses.npy", np.asarray(losses, dtype=np.float64))
    np.savez(args.output_dir / "logits.npz", **{f"step_{i}": value
                                                for i, value in enumerate(logits)})
    if args.save_selected_tensors:
        np.savez(args.output_dir / "selected_tensors.npz", **selected_arrays)
    report = {
        "schema_version": 1,
        "kind": "simcse.source-training-probe",
        "source_commit": "49fa580a853752ede55b8c76d9debf748e214d3f",
        "base_checkpoint_sha256": BASE_SHA256,
        "batch_manifest_sha256": file_sha256(args.batch_dir / "manifest.json"),
        "seed": SEED,
        "steps": STEPS,
        "dropout": args.dropout,
        "precision": args.precision,
        "pooler_type": "cls",
        "temperature": 0.05,
        "optimizer": {"name": "torch.optim.AdamW", "lr": LR, "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": 0.0, "foreach": False},
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_device": torch.cuda.get_device_name(),
        "loading": loading,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_state_sha256": initial_digest,
        "step_reports": steps,
        "losses_sha256": file_sha256(args.output_dir / "losses.npy"),
        "logits_sha256": file_sha256(args.output_dir / "logits.npz"),
        "selected_tensors_sha256": (
            file_sha256(args.output_dir / "selected_tensors.npz")
            if args.save_selected_tensors else None
        ),
        "initial_state_archive_sha256": (
            file_sha256(args.output_dir / "train_initial.npz")
            if args.save_initial_state else None
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"losses": losses, "grad_norms": [x["grad_norm"] for x in steps],
                      "initial_state_sha256": initial_digest,
                      "output_dir": str(args.output_dir)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
