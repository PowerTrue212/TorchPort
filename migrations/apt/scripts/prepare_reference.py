from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from common import ROOT, continual_metrics, digest, load_config, tree_digest, write_json


def version_matches(actual: str, expected: str) -> bool:
    return actual == expected or actual.startswith(expected + "+")


def check_environment(config: dict) -> dict:
    import timm
    import torch
    import torchvision

    expected = config["environment"]
    actual = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    expected_versions = {
        "python": expected["source_python"],
        "pytorch": expected["pytorch"],
        "torchvision": expected["torchvision"],
        "timm": expected["timm"],
        "numpy": expected["numpy"],
    }
    for key, expected_version in expected_versions.items():
        if not version_matches(str(actual[key]), str(expected_version)):
            raise RuntimeError(f"expected {key} {expected_version}, got {actual[key]}")
    if not torch.cuda.is_available() or "V100" not in str(actual["gpu"]).upper():
        raise RuntimeError(f"frozen protocol requires NVIDIA V100, got {actual['gpu']}")
    return actual


def prepare_reference_repo(source: Path, output: Path) -> Path:
    repo = output / "repo"
    if repo.exists():
        if tree_digest(repo) != tree_digest(source):
            raise RuntimeError("PyTorch reference source changed")
        return repo
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, repo, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return repo


def main() -> int:
    config = load_config()
    environment = check_environment(config)
    source = ROOT / "source" / "apt"
    data_root = ROOT / "shared" / "data"
    pretrained = ROOT / "shared" / "checkpoints" / config["backbone"]["checkpoint_filename"]
    if not source.is_dir() or not data_root.is_dir() or not pretrained.is_file():
        raise RuntimeError("run run_prepare.sh first")

    reference_root = ROOT / "reference" / "pytorch"
    repo = prepare_reference_repo(source, reference_root)
    run_dir = reference_root / "run"
    checkpoint = run_dir / "models" / "repeat-1" / "task-10" / "class.pth"
    env = os.environ.copy()
    env["APT_VIT_PRETRAINED"] = str(pretrained)
    env.setdefault("PYTHONHASHSEED", str(config["training"]["seed"]))

    training_complete = checkpoint.is_file() and (run_dir / "results-acc" / "pt.yaml").is_file()
    if not training_complete:
        command = [
            sys.executable,
            "-u",
            "run.py",
            "--config",
            "configs/cifar-100_prompt.yaml",
            "--gpuid",
            str(config["execution"]["gpu_index"]),
            "--repeat",
            "1",
            "--overwrite",
            "0",
            "--learner_type",
            "prompt",
            "--learner_name",
            "APT_Learner",
            "--prompt_param",
            str(config["training"]["prompt_dropout"]),
            "--lr",
            str(config["training"]["learning_rate"]),
            "--seed",
            str(config["training"]["seed"]),
            "--ema_coeff",
            str(config["training"]["ema_coefficient"]),
            "--schedule",
            str(config["training"]["epochs_per_task"]),
            "--batch_size",
            str(config["training"]["batch_size"]),
            "--log_dir",
            str(run_dir),
        ]
        # The original CLI does not expose dataroot; a reference-only symlink keeps
        # the source tree and all method workspaces identical.
        data_link = repo / "data"
        if not data_link.exists():
            data_link.symlink_to(data_root, target_is_directory=True)
        subprocess.run(command, cwd=repo, env=env, check=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    inference_dir = reference_root / "inference"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "reference_infer.py"),
            "--source",
            str(repo),
            "--data-root",
            str(data_root),
            "--checkpoint",
            str(checkpoint),
            "--pretrained",
            str(pretrained),
            "--output",
            str(inference_dir),
            "--gpu",
            str(config["execution"]["gpu_index"]),
        ],
        cwd=repo,
        env=env,
        check=True,
    )

    pt_yaml = yaml.safe_load((run_dir / "results-acc" / "pt.yaml").read_text(encoding="utf-8"))
    raw_matrix = np.asarray(pt_yaml["mean"], dtype=np.float32)
    if raw_matrix.shape != (10, 10):
        raise RuntimeError(f"unexpected PyTorch accuracy matrix: {raw_matrix.shape}")
    matrix = raw_matrix.T.copy()
    matrix[np.triu_indices(10, 1)] = np.nan

    inference = json.loads(
        (inference_dir / "inference_summary.json").read_text(encoding="utf-8")
    )
    final_task = np.asarray(inference["final_task_accuracies_percent"], dtype=np.float32)
    if not np.allclose(matrix[-1], final_task, rtol=0, atol=1e-5):
        raise RuntimeError("final inference does not match the training accuracy matrix")
    metrics = continual_metrics(matrix)

    shared = ROOT / "shared" / "reference"
    shared.mkdir(parents=True, exist_ok=True)
    np.save(shared / "accuracy_matrix.npy", matrix, allow_pickle=False)
    shutil.copy2(inference_dir / "predictions.npz", shared / "predictions.npz")
    summary = {
        "protocol_version": config["protocol_version"],
        "framework": "pytorch",
        **metrics,
        "final_task_accuracies_percent": final_task.tolist(),
        "final_accuracy_percent": float(inference["final_accuracy_percent"]),
        "test_samples": int(inference["test_samples"]),
        "accuracy_matrix_shape": list(matrix.shape),
    }
    write_json(shared / "summary.json", summary)

    relative_files = [
        "shared/reference/accuracy_matrix.npy",
        "shared/reference/predictions.npz",
        "shared/reference/summary.json",
        str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
    ]
    files = {
        relative: {"sha256": digest(ROOT / relative), "bytes": (ROOT / relative).stat().st_size}
        for relative in relative_files
    }
    write_json(
        ROOT / "shared" / "provenance" / "reference.json",
        {
            "protocol_version": config["protocol_version"],
            "environment": environment,
            "source_tree_sha256": tree_digest(source),
            "summary": summary,
            "files": files,
        },
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
