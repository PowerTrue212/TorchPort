"""Write a controller-only supervised SimCSE training TaskSpec (training GATE v2).

Full-parameter update sketch (`sketch_dim`), 16 NLI triplet steps with the official
linear decay, and the derivable `position_ids` buffer excluded on both sides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = Path("D:/科研/SimCSE-Benchmark-v1")
SOURCE = BENCH / "source/SimCSE"
INPUTS = BENCH / "assets/train-sup-steps-v2"
# BERT-base 初始化 + 固定种子 MLP；与有监督源端探针的初始状态摘要一致
INITIAL = (BENCH / "controller/prep/train-source-tensors-evidence"
           / "train-source-tensors-r0/train_initial.npz")
INITIAL_SHA256 = "57a2f10c6a78ee6eb8f762a2620ff4f691eecc3841deb85d498b480675dc81c5"
INPUT_MANIFEST_SHA256 = "a367009e6689604aa8b3fe70f9d22e06e5dce42a4cbd4d3476cb003cbfce2df4"
STEPS, BATCH, SLOTS, LENGTH = 16, 8, 3, 32
SKETCH_DIM = 16384


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
    parser.add_argument("--task-id", default="simcse-sup-bert-base-train-step-v2")
    parser.add_argument("--jittor-home", default="/root/SimCSE-Benchmark-v1/jittor-cache-train")
    args = parser.parse_args()
    target = args.target_repo.resolve()
    for name in ("run.py", "train_config.json"):
        if not (target / name).is_file():
            raise ValueError(f"Target checkout has no {name}")
    if sha256(INITIAL) != INITIAL_SHA256:
        raise ValueError("Initial state archive differs")
    manifest_path = INPUTS / "manifest.json"
    if sha256(manifest_path) != INPUT_MANIFEST_SHA256:
        raise ValueError("Frozen input manifest differs")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    for name, info in manifest["files"].items():
        if sha256(INPUTS / f"{name}.npy") != info["sha256"]:
            raise ValueError(f"Frozen input differs: {name}")
        if info["shape"] != [STEPS, BATCH, SLOTS, LENGTH]:
            raise ValueError(f"Frozen input shape differs: {name}")
    for name, controlled in (("torchport_sup_train.py", HERE / "sup_train_source_entry.py"),
                             ("train_config.json", HERE / "train_config.json")):
        if sha256(SOURCE / name) != sha256(controlled):
            raise ValueError(f"Source adapter differs: {name}")
    task = {
        "schema_version": 1,
        "task_id": args.task_id,
        "source": {"repo": str(SOURCE), "entry": "torchport_sup_train.py",
                   "model": "load_train_model", "device": "cuda"},
        "target": {"repo": str(target), "entry": "run.py",
                   "model": "load_train_model", "device": "cuda",
                   "env": {"JITTOR_HOME": args.jittor_home,
                           "use_mkl": "0", "use_cutt": "0"}},
        "inputs": [
            {"shape": [STEPS, BATCH, SLOTS, LENGTH], "dtype": "int64",
             "path": str(INPUTS / f"{name}.npy")}
            for name in ("input_ids", "attention_mask", "token_type_ids")
        ],
        "weights": str(INITIAL),
        "seed": 20261027,
        "protocol": {"profile": "train-step", "repeats": 2, "timeout": 5400,
                     "pass_k": 8.0,
                     "ignore_state": ["embeddings.position_ids"],
                     "train": {"optimizer": "build_optimizer", "step": "train_step",
                               "sketch_dim": SKETCH_DIM}},
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"task": str(args.output), "target": str(target),
                      "steps": STEPS, "sketch_dim": SKETCH_DIM}, ensure_ascii=True))


if __name__ == "__main__":
    main()
