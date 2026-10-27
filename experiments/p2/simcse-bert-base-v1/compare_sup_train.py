"""Check repeatability and the FP64 floor of the supervised SimCSE source probes.

Inputs are three probe_sup_train_source.py output directories: two FP32 runs and
one FP64 run. The FP64 differences describe the numerical floor only; they do not
set a target tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(directory: Path) -> dict:
    report = json.loads((directory / "report.json").read_text("utf-8"))
    for name, key in (("losses.npy", "losses_sha256"), ("logits.npz", "logits_sha256"),
                      ("selected_tensors.npz", "selected_tensors_sha256")):
        if sha256(directory / name) != report[key]:
            raise ValueError(f"Evidence checksum failed: {directory / name}")
    return report


def archives_equal(left: Path, right: Path) -> bool:
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        return a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)


def error(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape:
        raise ValueError(f"Tensor shapes differ: {left.shape} and {right.shape}")
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"max_abs": float(difference.max()),
            "p99_abs": float(np.percentile(difference, 99))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-a", type=Path, required=True)
    parser.add_argument("--fp32-b", type=Path, required=True)
    parser.add_argument("--fp64", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first, second, wide = load(args.fp32_a), load(args.fp32_b), load(args.fp64)

    repeat = {
        "losses_bitwise_equal": bool(np.array_equal(
            np.load(args.fp32_a / "losses.npy"), np.load(args.fp32_b / "losses.npy"))),
        "logits_bitwise_equal": archives_equal(args.fp32_a / "logits.npz",
                                               args.fp32_b / "logits.npz"),
        "selected_tensors_bitwise_equal": archives_equal(
            args.fp32_a / "selected_tensors.npz", args.fp32_b / "selected_tensors.npz"),
        "model_state_digests_equal": (
            first["initial_state_sha256"] == second["initial_state_sha256"]
            and [s["state_sha256_after_step"] for s in first["step_reports"]]
            == [s["state_sha256_after_step"] for s in second["step_reports"]]),
    }
    repeat["passed"] = all(repeat.values())

    floor = {"losses": error(np.load(args.fp32_a / "losses.npy"),
                             np.load(args.fp64 / "losses.npy"))}
    with np.load(args.fp32_a / "logits.npz") as a, np.load(args.fp64 / "logits.npz") as b:
        floor["logits"] = {k: error(a[k], b[k]) for k in a.files}
    with (np.load(args.fp32_a / "selected_tensors.npz") as a,
          np.load(args.fp64 / "selected_tensors.npz") as b):
        floor["selected_tensors"] = {k: error(a[k], b[k]) for k in a.files}

    result = {
        "schema_version": 1,
        "losses_fp32": [s["loss"] for s in first["step_reports"]],
        "losses_fp64": [s["loss"] for s in wide["step_reports"]],
        "grad_norms_fp32": [s["grad_norm"] for s in first["step_reports"]],
        "repeatability": repeat,
        "fp64_floor": floor,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "losses_fp32": result["losses_fp32"],
        "grad_norms_fp32": result["grad_norms_fp32"],
        "repeat_passed": repeat["passed"],
        "loss_floor": floor["losses"]["max_abs"],
        "logit_floor": max(v["max_abs"] for v in floor["logits"].values()),
        "grad_floor": max(v["max_abs"] for k, v in floor["selected_tensors"].items()
                          if ".grad." in k),
    }))
    if not repeat["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
