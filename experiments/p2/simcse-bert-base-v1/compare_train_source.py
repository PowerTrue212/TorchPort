"""Check repeatability of the two fixed SimCSE source-training probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_state_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as state:
        for name in sorted(state.files):
            array = state[name]
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(str(array.dtype).encode("ascii") + b"\0")
            digest.update(str(array.shape).encode("ascii") + b"\0")
            digest.update(array.tobytes())
    return digest.hexdigest()


def compare_pair(root: Path, prefix: str) -> dict:
    first = root / f"{prefix}-r0"
    second = root / f"{prefix}-r1"
    reports = []
    for directory in (first, second):
        report = json.loads((directory / "report.json").read_text("utf-8"))
        for name, key in (("losses.npy", "losses_sha256"),
                          ("logits.npz", "logits_sha256")):
            if sha256(directory / name) != report[key]:
                raise ValueError(f"Evidence checksum failed: {directory / name}")
        reports.append(report)

    losses = [np.load(directory / "losses.npy", allow_pickle=False)
              for directory in (first, second)]
    with np.load(first / "logits.npz", allow_pickle=False) as left:
        with np.load(second / "logits.npz", allow_pickle=False) as right:
            logits_equal = left.files == right.files and all(
                np.array_equal(left[key], right[key]) for key in left.files
            )
    state_equal = (
        reports[0]["initial_state_sha256"] == reports[1]["initial_state_sha256"]
        and [step["state_sha256_after_step"] for step in reports[0]["step_reports"]]
        == [step["state_sha256_after_step"] for step in reports[1]["step_reports"]]
    )
    gradients_equal = [step["grad_norm"] for step in reports[0]["step_reports"]] == [
        step["grad_norm"] for step in reports[1]["step_reports"]
    ]
    result = {
        "dropout": reports[0]["dropout"],
        "losses": losses[0].tolist(),
        "losses_bitwise_equal": bool(np.array_equal(losses[0], losses[1])),
        "logits_bitwise_equal": bool(logits_equal),
        "model_state_digests_equal": state_equal,
        "gradient_norms_equal": gradients_equal,
    }
    result["passed"] = all(
        result[key] for key in ("losses_bitwise_equal", "logits_bitwise_equal",
                                "model_state_digests_equal", "gradient_norms_equal")
    )
    return result


def compare_tensor_pair(root: Path) -> dict:
    prefix = "train-source-tensors"
    result = compare_pair(root, prefix)
    first = root / f"{prefix}-r0"
    second = root / f"{prefix}-r1"
    reports = [json.loads((directory / "report.json").read_text("utf-8"))
               for directory in (first, second)]
    selected_equal = True
    with np.load(first / "selected_tensors.npz", allow_pickle=False) as left:
        with np.load(second / "selected_tensors.npz", allow_pickle=False) as right:
            selected_equal = left.files == right.files and all(
                np.array_equal(left[name], right[name]) for name in left.files
            )
    selected_hashes_valid = all(
        sha256(directory / "selected_tensors.npz") == report["selected_tensors_sha256"]
        for directory, report in zip((first, second), reports, strict=True)
    )
    initial_path = first / "train_initial.npz"
    initial_archive_valid = (
        sha256(initial_path) == reports[0]["initial_state_archive_sha256"]
        and archive_state_sha256(initial_path) == reports[0]["initial_state_sha256"]
    )
    result.update({
        "selected_tensors_bitwise_equal": selected_equal,
        "selected_tensors_hashes_valid": selected_hashes_valid,
        "initial_archive_matches_state_digest": initial_archive_valid,
    })
    result["passed"] = result["passed"] and all((
        selected_equal, selected_hashes_valid, initial_archive_valid,
    ))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if (args.root / "train-source-tensors-r0").exists():
        result = {"selected_tensor_runs": compare_tensor_pair(args.root)}
    else:
        result = {
            "no_dropout": compare_pair(args.root, "train-source"),
            "dropout_0_1": compare_pair(args.root, "train-source-dropout"),
        }
    print(json.dumps(result, indent=2))
    if not all(part["passed"] for part in result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
