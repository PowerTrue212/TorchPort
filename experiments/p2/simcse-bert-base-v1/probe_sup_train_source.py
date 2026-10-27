"""Run a small deterministic source-only supervised SimCSE training check.

Follows the official supervised recipe (BERT-base init, cls pooler with MLP in
training and evaluation, temperature 0.05, NLI triplets with a hard negative),
with dropout 0 so a target can be compared without replaying dropout masks.
Each step consumes its own frozen batch from prepare_sup_train_batch.py.
"""

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
LR = 5e-5  # 官方有监督脚本的学习率
TEMPERATURE = 0.05
NAMES = ("input_ids", "attention_mask", "token_type_ids")
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
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
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
    from transformers import __version__ as transformers_version  # noqa: PLC0415

    sys.path.insert(0, str(args.source_repo.resolve()))
    from simcse.models import BertForCL  # noqa: PLC0415

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    config = BertConfig.from_json_file(str(args.base_dir / "config.json"))
    config.hidden_dropout_prob = 0.0
    config.attention_probs_dropout_prob = 0.0
    model_args = SimpleNamespace(pooler_type="cls", temp=TEMPERATURE, do_mlm=False,
                                 mlp_only_train=False, hard_negative_weight=0.0)
    model, loading = BertForCL.from_pretrained(
        str(args.base_dir), config=config, model_args=model_args,
        local_files_only=True, output_loading_info=True,
    )
    model.cuda().train()
    if args.precision == "fp64":
        model.double()
    arrays = {name: np.load(args.batch_dir / f"{name}.npy", allow_pickle=False)
              for name in NAMES}
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 4 or next(iter(shapes))[2] != 3:
        raise ValueError(f"Frozen batches must be [steps, batch, 3, length]: {shapes}")
    steps_count, batch = next(iter(shapes))[:2]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.0, foreach=False)
    parameters = dict(model.named_parameters())
    for name in SELECTED:
        if name not in parameters:
            raise KeyError(name)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected = {f"initial.param.{name}": parameters[name].detach().cpu().numpy().copy()
                for name in SELECTED}
    losses, logits, reports = [], [], []
    initial_digest = state_sha256(model)
    for index in range(steps_count):
        inputs = {name: torch.from_numpy(value[index]).cuda() for name, value in arrays.items()}
        optimizer.zero_grad(set_to_none=True)
        similarity = []
        hook = model.sim.register_forward_hook(
            lambda _module, _inputs, result: similarity.append(result.detach().cpu().numpy())
        )
        try:
            output = model(**inputs, return_dict=True)
        finally:
            hook.remove()
        # 有监督：一次正样本相似度 (batch, batch) + 一次困难负样本相似度 (batch, batch)
        if len(similarity) != 2 or any(item.shape != (batch, batch) for item in similarity):
            raise ValueError(f"Unexpected similarity outputs at step {index}")
        loss = output.loss
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at step {index}")
        losses.append(float(loss.detach().cpu()))
        logits.append(np.concatenate(similarity, axis=1))
        loss.backward()
        grad_norm = float(torch.stack([
            p.grad.detach().double().square().sum() for p in model.parameters()
            if p.grad is not None
        ]).sum().sqrt().cpu())
        if not np.isfinite(grad_norm) or grad_norm <= 0:
            raise ValueError(f"Missing or non-finite gradient at step {index}")
        selected.update({
            f"step_{index}.grad.{name}": parameters[name].grad.detach().cpu().numpy().copy()
            for name in SELECTED
        })
        optimizer.step()
        selected.update({
            f"step_{index}.param.{name}": parameters[name].detach().cpu().numpy().copy()
            for name in SELECTED
        })
        reports.append({"index": index, "loss": losses[-1], "grad_norm": grad_norm,
                        "state_sha256_after_step": state_sha256(model)})

    np.save(args.output_dir / "losses.npy", np.asarray(losses, dtype=np.float64))
    np.savez(args.output_dir / "logits.npz",
             **{f"step_{i}": value for i, value in enumerate(logits)})
    np.savez(args.output_dir / "selected_tensors.npz", **selected)
    report = {
        "schema_version": 1,
        "kind": "simcse.supervised-source-training-probe",
        "source_commit": "49fa580a853752ede55b8c76d9debf748e214d3f",
        "base_checkpoint_sha256": BASE_SHA256,
        "batch_manifest_sha256": file_sha256(args.batch_dir / "manifest.json"),
        "seed": SEED,
        "steps": int(steps_count),
        "batch": int(batch),
        "dropout": 0.0,
        "precision": args.precision,
        "pooler_type": "cls",
        "mlp_only_train": False,
        "temperature": TEMPERATURE,
        "hard_negative_weight": 0.0,
        "optimizer": {"name": "torch.optim.AdamW", "lr": LR, "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": 0.0, "foreach": False},
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_device": torch.cuda.get_device_name(),
        "loading": loading,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_state_sha256": initial_digest,
        "step_reports": reports,
        "losses_sha256": file_sha256(args.output_dir / "losses.npy"),
        "logits_sha256": file_sha256(args.output_dir / "logits.npz"),
        "selected_tensors_sha256": file_sha256(args.output_dir / "selected_tensors.npz"),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"losses": losses, "grad_norms": [r["grad_norm"] for r in reports],
                      "initial_state_sha256": initial_digest}))


if __name__ == "__main__":
    main()
