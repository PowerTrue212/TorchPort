from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final PyTorch APT checkpoint inference")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    os.environ["APT_VIT_PRETRAINED"] = str(args.pretrained.resolve())
    os.chdir(source)
    sys.path.insert(0, str(source))

    from run import get_args, seed_everything
    from trainer import Trainer

    protocol_args = get_args(
        [
            "--config",
            "configs/cifar-100_prompt.yaml",
            "--gpuid",
            str(args.gpu),
            "--repeat",
            "1",
            "--overwrite",
            "0",
            "--learner_type",
            "prompt",
            "--learner_name",
            "APT_Learner",
            "--prompt_param",
            "0.01",
            "--lr",
            "0.004",
            "--seed",
            "1",
            "--ema_coeff",
            "0.7",
            "--schedule",
            "30",
            "--batch_size",
            "64",
            "--log_dir",
            str(args.checkpoint.resolve().parents[3]),
        ]
    )
    protocol_args.dataroot = str(args.data_root.resolve())
    protocol_args.workers = 4
    seed_everything(protocol_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = Trainer(
        protocol_args,
        seed=protocol_args.seed,
        cur_iter=0,
        metric_keys=["acc", "time", "general_forgetting", "coda_forgetting"],
        save_keys=["global", "pt"],
    )
    learner = trainer.learner
    learner.add_valid_output_dim(sum(len(task) for task in trainer.tasks_logits))
    learner.task_count = trainer.max_task - 1
    learner.pre_steps()
    learner.load_model(str(args.checkpoint.resolve().parent) + os.sep)
    model = learner.model
    try:
        model.module.task_id = trainer.max_task - 1
    except AttributeError:
        model.task_id = trainer.max_task - 1

    logits_parts = []
    target_parts = []
    prediction_parts = []
    task_id_parts = []
    task_accuracies = []
    model.eval()
    with torch.inference_mode():
        for task_index in range(trainer.max_task):
            trainer.test_dataset.load_dataset(task_index, train=True)
            loader = DataLoader(
                trainer.test_dataset,
                batch_size=protocol_args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=protocol_args.workers,
            )
            correct = 0
            samples = 0
            for images, targets, _ in loader:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                logits = model(images)[:, : learner.valid_out_dim]
                predictions = logits.argmax(dim=1)
                correct += int((predictions == targets).sum().item())
                samples += int(targets.numel())
                logits_parts.append(logits.cpu().numpy().astype(np.float32, copy=False))
                target_parts.append(targets.cpu().numpy().astype(np.int64, copy=False))
                prediction_parts.append(predictions.cpu().numpy().astype(np.int64, copy=False))
                task_id_parts.append(np.full(targets.numel(), task_index, dtype=np.int64))
            task_accuracies.append(100.0 * correct / samples)

    logits = np.concatenate(logits_parts)
    targets = np.concatenate(target_parts)
    predictions = np.concatenate(prediction_parts)
    task_ids = np.concatenate(task_id_parts)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "predictions.npz",
        logits=logits,
        targets=targets,
        predictions=predictions,
        task_ids=task_ids,
    )
    summary = {
        "test_samples": int(targets.size),
        "final_accuracy_percent": 100.0 * float((predictions == targets).sum()) / targets.size,
        "final_task_accuracies_percent": task_accuracies,
        "logits_shape": list(logits.shape),
    }
    (args.output / "inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
