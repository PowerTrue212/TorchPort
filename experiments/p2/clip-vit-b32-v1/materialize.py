"""Write a local pilot TaskSpec for the CLIP migration seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[3]
    source = project / "migrations/CLIP/source/openai-CLIP"
    source_entry = source / "torchport_adapter.py"
    source_template = Path(__file__).with_name("source_adapter.py")
    if not source_entry.exists():
        source_entry.write_bytes(source_template.read_bytes())
    elif source_entry.read_bytes() != source_template.read_bytes():
        raise ValueError("Source adapter differs from the frozen template")
    data = project / "migrations/CLIP/data/pilot"
    weights = project / "migrations/CLIP/checkpoint/weights_fp32.npz"
    candidate = args.candidate.resolve()
    required = [
        source_entry, candidate / "evaluation_entry.py",
        data / "images.npy", data / "tokens.npy", weights,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing pilot materials: " + ", ".join(missing))
    task = {
        "schema_version": 1,
        "task_id": "clip-vit-b32-cifar10-pilot-forward-v1",
        "source": {
            "repo": str(source), "entry": "torchport_adapter.py",
            "model": "TorchPortCLIP", "device": "cuda",
        },
        "target": {
            "repo": str(candidate), "entry": "evaluation_entry.py",
            "model": "TorchPortCLIP", "device": "cuda",
        },
        "inputs": [
            {"shape": [100, 3, 224, 224], "dtype": "float32", "path": str(data / "images.npy")},
            {"shape": [10, 77], "dtype": "int64", "path": str(data / "tokens.npy")},
        ],
        "weights": str(weights),
        "seed": 20261025,
        "protocol": {"profile": "fp32-inference", "repeats": 2, "timeout": 1800},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
