from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from common import (
    ROOT,
    continual_metrics,
    load_config,
    method_ids,
    read_json_lines,
    verify_reference_files,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one full Jittor APT migration")
    parser.add_argument("--method", required=True, choices=method_ids())
    return parser.parse_args()


def import_submission(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("apt_migration_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import submission: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run_experiment", None)):
        raise TypeError("submission must define callable run_experiment")
    return module


def validate_matrix(path: Path) -> np.ndarray:
    matrix = np.load(path, allow_pickle=False)
    if matrix.shape != (10, 10) or matrix.dtype != np.float32:
        raise RuntimeError(f"invalid accuracy matrix: {matrix.shape} {matrix.dtype}")
    for row in range(10):
        if not np.isfinite(matrix[row, : row + 1]).all():
            raise RuntimeError(f"non-finite evaluated accuracy at row {row}")
        if not np.isnan(matrix[row, row + 1 :]).all():
            raise RuntimeError(f"unused upper triangle must be NaN at row {row}")
    finite = matrix[np.isfinite(matrix)]
    if (finite < 0).any() or (finite > 100).any():
        raise RuntimeError("accuracy matrix values must be percentages in [0, 100]")
    return matrix


def validate_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    package = np.load(path, allow_pickle=False)
    required = {"logits", "targets", "predictions", "task_ids"}
    if set(package.files) != required:
        raise RuntimeError(f"prediction archive keys must be {sorted(required)}")
    logits = package["logits"]
    targets = package["targets"]
    predictions = package["predictions"]
    task_ids = package["task_ids"]
    if logits.shape != (10000, 100) or logits.dtype != np.float32:
        raise RuntimeError(f"invalid logits: {logits.shape} {logits.dtype}")
    if not np.isfinite(logits).all():
        raise RuntimeError("final logits contain NaN or infinity")
    for name, value in {
        "targets": targets,
        "predictions": predictions,
        "task_ids": task_ids,
    }.items():
        if value.shape != (10000,) or value.dtype != np.int64:
            raise RuntimeError(f"invalid {name}: {value.shape} {value.dtype}")
    if (targets < 0).any() or (targets >= 100).any():
        raise RuntimeError("targets are outside [0, 99]")
    if not np.array_equal(predictions, logits.argmax(axis=1).astype(np.int64)):
        raise RuntimeError("predictions do not equal argmax(logits)")
    counts = np.bincount(task_ids, minlength=10)
    if counts.shape != (10,) or not np.array_equal(counts, np.full(10, 1000)):
        raise RuntimeError(f"each task must contribute exactly 1000 test images: {counts}")
    return logits, targets, predictions, task_ids


def main() -> int:
    args = parse_args()
    config = load_config()
    reference_manifest = verify_reference_files()
    reference_summary = reference_manifest["summary"]
    rounds = read_json_lines(ROOT / "results" / args.method / "rounds.jsonl")
    maximum = int(config["execution"]["max_repair_rounds"])
    if len(rounds) > maximum:
        raise RuntimeError(f"repair log exceeds frozen maximum {maximum}")

    result = {
        "protocol_version": config["protocol_version"],
        "method": args.method,
        "rn": False,
        "average_accuracy_percent": None,
        "forgetting_percent": None,
        "delta_accuracy_pp": None,
        "final_accuracy_percent": None,
        "seen_accuracy_percent": None,
        "rounds": len(rounds),
        "failure": None,
        "elapsed_seconds": None,
    }
    result_dir = ROOT / "results" / args.method
    run_dir = result_dir / "run"
    attempt_dir = result_dir / "attempts"
    run_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    try:
        import jittor as jt

        if jt.__version__ != config["environment"]["jittor"]:
            raise RuntimeError(
                f"expected Jittor {config['environment']['jittor']}, got {jt.__version__}"
            )
        if np.__version__ != config["environment"]["numpy"]:
            raise RuntimeError(
                f"expected NumPy {config['environment']['numpy']}, got {np.__version__}"
            )
        gpu_names = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()
        if not bool(jt.has_cuda) or not any("V100" in name.upper() for name in gpu_names):
            raise RuntimeError(f"frozen protocol requires Jittor CUDA on V100, got {gpu_names}")
        jt.flags.use_cuda = 1
        result["gpu_names"] = gpu_names
        result["jittor_version"] = jt.__version__

        submission_path = ROOT / "workspaces" / args.method / "repo" / "submission.py"
        if not submission_path.is_file():
            raise FileNotFoundError(submission_path)
        torch_was_loaded = "torch" in sys.modules
        submission = import_submission(submission_path)
        if not torch_was_loaded and "torch" in sys.modules:
            raise RuntimeError("submission imported PyTorch in the Jittor evaluator")

        returned = submission.run_experiment(
            protocol=config,
            data_root=ROOT / "shared" / "data",
            pretrained_path=(
                ROOT / "shared" / "checkpoints" / config["backbone"]["checkpoint_filename"]
            ),
            output_dir=run_dir,
        )
        if not torch_was_loaded and "torch" in sys.modules:
            raise RuntimeError("submission executed PyTorch in the Jittor evaluator")
        if not isinstance(returned, dict):
            raise TypeError("run_experiment must return a dictionary")
        json.dumps(returned, ensure_ascii=False)
        if returned.get("framework") != "jittor" or returned.get("test_samples") != 10000:
            raise RuntimeError(f"invalid submission summary: {returned}")

        matrix = validate_matrix(run_dir / "accuracy_matrix.npy")
        _, targets, predictions, task_ids = validate_predictions(run_dir / "predictions.npz")
        final_task_accuracies = []
        for task_index in range(10):
            mask = task_ids == task_index
            final_task_accuracies.append(
                100.0 * float((predictions[mask] == targets[mask]).sum()) / int(mask.sum())
            )
        if not np.allclose(matrix[-1], final_task_accuracies, rtol=0, atol=1e-5):
            raise RuntimeError("final predictions do not match the last accuracy-matrix row")

        metrics = continual_metrics(matrix)
        average_accuracy = metrics["average_accuracy_percent"]
        final_accuracy = 100.0 * float((predictions == targets).sum()) / targets.size
        if not np.isclose(average_accuracy, final_accuracy, rtol=0, atol=1e-5):
            raise RuntimeError("final full-test accuracy does not match Average Accuracy")
        result.update(
            {
                "rn": True,
                "average_accuracy_percent": average_accuracy,
                "forgetting_percent": metrics["forgetting_percent"],
                "delta_accuracy_pp": average_accuracy
                - float(reference_summary["average_accuracy_percent"]),
                "final_accuracy_percent": final_accuracy,
                "seen_accuracy_percent": metrics["seen_accuracy_percent"],
            }
        )
    except Exception as error:
        result["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started

    attempt = f"attempt_{len(rounds):02d}"
    write_json(attempt_dir / f"{attempt}.json", result)
    write_json(result_dir / "result.json", result)
    if result["rn"]:
        shutil.copy2(run_dir / "accuracy_matrix.npy", result_dir / "accuracy_matrix.npy")
        shutil.copy2(run_dir / "predictions.npz", result_dir / "predictions.npz")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["rn"] else 2


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "1")
    raise SystemExit(main())
