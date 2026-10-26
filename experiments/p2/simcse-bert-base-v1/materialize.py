"""Write hidden SimCSE inference TaskSpecs for one clean candidate checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = Path("D:/科研/SimCSE-Benchmark-v1")
CHECKPOINT = BENCH / "assets/models/princeton-nlp--unsup-simcse-bert-base-uncased"
DATA = BENCH / "assets/stsb-test-l64"
WEIGHTS_SHA256 = "0b1a8d07d607d6c23660ea9b80ae729ab37ff30ee93f35db2c7e8a3afbe8a4bb"


def task(candidate: Path, *, smoke: bool) -> dict:
    prefix = "smoke_" if smoke else ""
    inputs = []
    for name in ("input_ids", "attention_mask", "token_type_ids"):
        path = DATA / f"{prefix}{name}.npy"
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected = (64, 64) if smoke else (2758, 64)
        if array.shape != expected or array.dtype != np.int64:
            raise ValueError(f"Unexpected frozen input: {path}")
        inputs.append({"shape": list(array.shape), "dtype": "int64", "path": str(path)})
    return {
        "schema_version": 1,
        "task_id": ("simcse-unsup-bert-base-stsb-smoke-forward-v1" if smoke
                    else "simcse-unsup-bert-base-stsb-forward-v1"),
        "source": {"repo": str(BENCH / "source/SimCSE"),
                   "entry": "torchport_adapter.py", "model": "TorchPortSimCSE",
                   "device": "cuda"},
        "target": {"repo": str(candidate), "entry": "evaluation_entry.py",
                   "model": "TorchPortSimCSE", "device": "cuda",
                   "env": {"JITTOR_HOME":
                           f"/root/SimCSE-Benchmark-v1/jittor-cache-{candidate.name}",
                           "use_mkl": "0", "use_cutt": "0"}},
        "inputs": inputs,
        "weights": str(CHECKPOINT / "weights_fp32.npz"),
        "seed": 20261026,
        "protocol": {"profile": "fp32-inference", "repeats": 2,
                     "timeout": 1800 if smoke else 3600},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("none", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate = BENCH / "workspaces" / args.condition
    source = BENCH / "source/SimCSE"
    if (source / "torchport_adapter.py").read_text(encoding="utf-8") != (
        HERE / "source_adapter.py"
    ).read_text(encoding="utf-8"):
        raise ValueError("Source adapter differs from the controlled template")
    if (candidate / "evaluation_entry.py").read_text(encoding="utf-8") != (
        HERE / "candidate_entry.py"
    ).read_text(encoding="utf-8"):
        raise ValueError("Candidate entry differs from the controlled template")
    if not (candidate / "run.py").is_file():
        raise FileNotFoundError(candidate / "run.py")
    weights = CHECKPOINT / "weights_fp32.npz"
    with weights.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != WEIGHTS_SHA256:
            raise ValueError("Frozen weight archive SHA-256 differs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for smoke, label in ((True, "smoke"), (False, "stsb")):
        output = args.output_dir / f"task-{label}-{args.condition}.json"
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(task(candidate, smoke=smoke), ensure_ascii=False,
                                     indent=2) + "\n", encoding="utf-8")
        print(output.resolve())


if __name__ == "__main__":
    main()
