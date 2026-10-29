"""Create a controller-only LightGCN two-step TaskSpec and empty target seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, inputs, target = (p.resolve() for p in (args.source, args.inputs, args.target))
    if args.output.exists() or target.exists():
        raise FileExistsError(args.output if args.output.exists() else target)
    if sha256(source / "torchport_train_adapter.py") != sha256(HERE / "train_source_adapter.py"):
        raise ValueError("materialized source adapter differs")
    manifest = json.loads((inputs / "manifest.json").read_text("utf-8"))
    for name, expected in manifest["files"].items():
        if sha256(inputs / name) != expected:
            raise ValueError(f"pilot input changed: {name}")
    target.mkdir(parents=True)
    (target / "evaluation_entry.py").write_text(
        '"""LightGCN training target seed; migration agent implements this entry."""\n'
        "def load_train_model():\n"
        '    raise NotImplementedError("Jittor LightGCN training migration is not implemented")\n',
        encoding="utf-8",
    )
    task = {
        "schema_version": 1,
        "task_id": "lightgcn-lastfm-train-two-step-pilot-r1",
        "source": {
            "repo": str(source), "entry": "torchport_train_adapter.py",
            "model": "load_train_model", "device": "cuda",
            "env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "TP_LIGHTGCN_DEPS": "/root/data-fs/LightGCN-Benchmark-v1/deps",
                    "HOME": "/root/data-fs/LightGCN-Benchmark-v1/home"},
        },
        "target": {
            "repo": str(target), "entry": "evaluation_entry.py",
            "model": "load_train_model", "device": "cuda",
            "env": {"HOME": "/root/data-fs/LightGCN-Benchmark-v1/home",
                    "JITTOR_HOME": "/root/data-fs/LightGCN-Benchmark-v1/jittor-cache",
                    "use_mkl": "0", "use_cutt": "0"},
        },
        "inputs": [
            {"shape": [2, 4], "dtype": "int64", "path": str(inputs / name)}
            for name in ("users.npy", "positive_items.npy", "negative_items.npy")
        ],
        "weights": str(inputs / "initial-weights.npz"),
        "seed": 20261029,
        "protocol": {
            "profile": "train-step", "repeats": 1, "timeout": 1200,
            "pass_k": 8.0,
            "train": {"optimizer": "build_optimizer", "step": "train_step",
                      "sketch_dim": 4096},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"task": str(args.output), "seed": str(target),
                      "adapter_sha256": sha256(source / "torchport_train_adapter.py")}))


if __name__ == "__main__":
    main()
