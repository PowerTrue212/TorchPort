"""Write a controller-only CLIP train-step feasibility TaskSpec.

This four-step pilot is for source oracle and target reference calibration. It
is not a frozen Benchmark v1 Agent task or a claim of training parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CLIP = ROOT / "migrations/CLIP"
SOURCE = CLIP / "source/openai-CLIP"
INPUTS = CLIP / "data/train-pilot"
WEIGHTS = CLIP / "checkpoint/weights_fp32.npz"
MANIFEST_SHA256 = "0ef29208babcb1f4836fa79d4b1b2d2d589dbfcdba368cbb2e96f26468602a1f"
ADAPTER_SHA256 = "5103a3319dc96b6857005874802e54e1ce4e7fc019b5b88e09d0c3be1cc4309c"
WEIGHTS_SHA256 = "d89c2d139d6398b23dceef40ec926a20ee72a986e0763101fdf366f27c9cab5d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.target_repo.resolve()
    if not (target / "evaluation_entry.py").is_file():
        raise FileNotFoundError("CLIP target seed entry is missing")
    if sha256(INPUTS / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("CLIP training input manifest changed")
    manifest = json.loads((INPUTS / "manifest.json").read_text("utf-8"))
    for name, expected in manifest["output_sha256"].items():
        if sha256(INPUTS / name) != expected:
            raise ValueError(f"CLIP training input changed: {name}")
    if sha256(WEIGHTS) != WEIGHTS_SHA256:
        raise ValueError("Official CLIP checkpoint changed")
    controlled = HERE / "train_source_adapter.py"
    source_entry = SOURCE / "torchport_train_adapter.py"
    if sha256(controlled) != ADAPTER_SHA256:
        raise ValueError("Controlled CLIP training adapter changed")
    if source_entry.exists():
        if sha256(source_entry) != ADAPTER_SHA256:
            raise ValueError("Materialized CLIP training adapter differs")
    else:
        source_entry.write_bytes(controlled.read_bytes())
    task = {
        "schema_version": 1,
        "task_id": "clip-vit-b32-cifar10-train-pilot-v2",
        "source": {
            "repo": str(SOURCE), "entry": source_entry.name,
            "model": "load_train_model", "device": "cuda",
            "env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "HOME": "/root/data-fs/torchport-clip-train-pilot"},
        },
        "target": {
            "repo": str(target), "entry": "evaluation_entry.py",
            "model": "load_train_model", "device": "cuda",
            "env": {"JITTOR_HOME": "/root/data-fs/torchport-clip-train-pilot/jittor-cache",
                    "HOME": "/root/data-fs/torchport-clip-train-pilot",
                    "use_mkl": "0", "use_cutt": "0"},
        },
        "inputs": [
            {"shape": [4, 10, 3, 224, 224], "dtype": "float32",
             "path": str(INPUTS / "images.npy")},
            {"shape": [4, 10, 77], "dtype": "int64",
             "path": str(INPUTS / "tokens.npy")},
        ],
        "weights": str(WEIGHTS),
        "seed": 20261028,
        "protocol": {
            "profile": "train-step", "repeats": 2, "timeout": 1800,
            "pass_k": 8.0,
            "train": {"optimizer": "build_optimizer", "step": "train_step",
                      "sketch_dim": 16384},
        },
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"task": str(args.output), "source": str(SOURCE),
                      "target": str(target), "steps": 4}, ensure_ascii=False))


if __name__ == "__main__":
    main()
