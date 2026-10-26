"""Write a controller-only SimCSE training TaskSpec for one candidate checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = Path("D:/科研/SimCSE-Benchmark-v1")
SOURCE = BENCH / "source/SimCSE"
INPUTS = BENCH / "assets/train-steps-v1"
INITIAL = (BENCH / "controller/prep/train-source-tensors-evidence"
           / "train-source-tensors-r0/train_initial.npz")
INITIAL_SHA256 = "57a2f10c6a78ee6eb8f762a2620ff4f691eecc3841deb85d498b480675dc81c5"
INPUT_MANIFEST_SHA256 = "11eca3e3b808194599e30b069453ae962857f412b49d30e6a973cfec72ce445a"
OBSERVE = HERE / "train-observe-v1.json"


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
    parser.add_argument("--task-id", default="simcse-bert-base-train-step-seed-v1")
    parser.add_argument("--jittor-home", default="/root/SimCSE-Benchmark-v1/jittor-cache-train")
    args = parser.parse_args()
    target = args.target_repo.resolve()
    if not (target / "run.py").is_file():
        raise ValueError("Target checkout has no run.py")
    if not (target / "train_config.json").is_file():
        raise ValueError("Target checkout has no frozen training config")
    if sha256(INITIAL) != INITIAL_SHA256:
        raise ValueError("Initial state archive differs")
    manifest_path = INPUTS / "manifest.json"
    if sha256(manifest_path) != INPUT_MANIFEST_SHA256:
        raise ValueError("Frozen input manifest differs")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    for name, info in manifest["files"].items():
        if sha256(INPUTS / f"{name}.npy") != info["sha256"]:
            raise ValueError(f"Frozen input differs: {name}")
    for name, source in (("torchport_train.py", HERE / "train_source_entry.py"),
                         ("train_config.json", HERE / "train_config.json")):
        if sha256(SOURCE / name) != sha256(source):
            raise ValueError(f"Source adapter differs: {name}")
    observed = json.loads(OBSERVE.read_text("utf-8"))["parameters"]
    if len(observed) != len(set(observed)) or not observed:
        raise ValueError("Training observation names are missing or repeated")
    with np.load(INITIAL, allow_pickle=False) as state:
        missing = sorted(set(observed) - set(state.files))
        observed_elements = sum(int(np.prod(state[name].shape)) for name in observed)
    if missing:
        raise ValueError(f"Training observation names are absent from the initial state: {missing}")
    task = {
        "schema_version": 1,
        "task_id": args.task_id,
        "source": {"repo": str(SOURCE), "entry": "torchport_train.py",
                   "model": "load_train_model", "device": "cuda"},
        "target": {"repo": str(target), "entry": "run.py",
                   "model": "load_train_model", "device": "cuda",
                   "env": {"JITTOR_HOME": args.jittor_home,
                           "use_mkl": "0", "use_cutt": "0"}},
        "inputs": [
            {"shape": [2, 8, 2, 32], "dtype": "int64",
             "path": str(INPUTS / f"{name}.npy")}
            for name in ("input_ids", "attention_mask", "token_type_ids")
        ],
        "weights": str(INITIAL),
        "seed": 20261027,
        "protocol": {"profile": "train-step", "repeats": 2, "timeout": 3600,
                     "pass_k": 8.0,
                     "train": {"optimizer": "build_optimizer", "step": "train_step",
                               "observe": observed}},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"task": str(args.output), "target": str(target),
                      "checkpoint_sha256": INITIAL_SHA256,
                      "observed_parameters": len(observed),
                      "observed_elements": observed_elements}, ensure_ascii=True))


if __name__ == "__main__":
    main()
