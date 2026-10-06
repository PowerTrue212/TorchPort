from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / "RAFT" / "results" / "dsh_torchport" / "traces"
DATA = ROOT / "experiments" / "data" / "tensors"
FIGURE = ROOT / "experiments" / "figures" / "fig_raft"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_rms(value: np.ndarray) -> np.ndarray:
    value64 = value[0].astype(np.float64, copy=False)
    return np.sqrt(np.mean(np.square(value64), axis=0)).astype(np.float32)


def endpoint(value: np.ndarray) -> np.ndarray:
    value64 = value[0].astype(np.float64, copy=False)
    return np.sqrt(np.sum(np.square(value64), axis=0))


def write_iteration_metrics(source: np.lib.npyio.NpzFile, target: np.lib.npyio.NpzFile) -> None:
    rows = []
    previous_source = np.zeros_like(source["flow_up_it01"])
    previous_target = np.zeros_like(target["flow_up_it01"])
    for iteration in range(1, 13):
        key = f"flow_up_it{iteration:02d}"
        source_flow = source[key]
        target_flow = target[key]
        source_update = endpoint(source_flow - previous_source)
        target_update = endpoint(target_flow - previous_target)
        framework_delta = endpoint(target_flow - source_flow)
        rows.append(
            {
                "iteration": iteration,
                "pytorch_update_mean": float(source_update.mean()),
                "jittor_update_mean": float(target_update.mean()),
                "framework_mean": float(framework_delta.mean()),
                "framework_p99": float(np.percentile(framework_delta, 99)),
                "framework_max": float(framework_delta.max()),
            }
        )
        previous_source = source_flow
        previous_target = target_flow
    with (FIGURE / "iteration_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tensor_alignment() -> None:
    strict = json.loads((TRACE / "compare_strict.json").read_text(encoding="utf-8"))
    fp32 = json.loads((TRACE / "compare_fp32.json").read_text(encoding="utf-8"))
    fp32_by_name = {item["name"]: item for item in fp32["tensors"]}
    fields = [
        "position",
        "name",
        "kind",
        "max_absolute_error",
        "mean_absolute_error",
        "p99_absolute_error",
        "strict_mismatched_elements",
        "strict_mismatch_ratio",
        "strict_passed",
        "fp32_passed",
    ]
    rows = []
    for item in strict["tensors"]:
        name = item["name"]
        if name.startswith("flow_up_it"):
            kind = "flow_up"
        elif name.startswith("flow_low_it"):
            kind = "flow_low"
        elif name.startswith("feature"):
            kind = "feature"
        elif name.startswith("input"):
            kind = "input"
        else:
            kind = "output"
        rows.append(
            {
                "position": item["position"],
                "name": name,
                "kind": kind,
                "max_absolute_error": item["max_absolute_error"],
                "mean_absolute_error": item["mean_absolute_error"],
                "p99_absolute_error": item["p99_absolute_error"],
                "strict_mismatched_elements": item["mismatched_elements"],
                "strict_mismatch_ratio": item["mismatch_ratio"],
                "strict_passed": item["passed"],
                "fp32_passed": fp32_by_name[name]["passed"],
            }
        )
    with (FIGURE / "tensor_alignment.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    source_path = TRACE / "source.npz"
    target_path = TRACE / "target.npz"
    if not source_path.is_file() or not target_path.is_file():
        raise FileNotFoundError("TorchPort source.npz and target.npz are required")
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURE.mkdir(parents=True, exist_ok=True)
    with np.load(source_path, allow_pickle=False) as source, np.load(target_path, allow_pickle=False) as target:
        arrays: dict[str, np.ndarray] = {}
        for index in (0, 1):
            key = f"feature.{index}"
            source_map = feature_rms(source[key])
            target_map = feature_rms(target[key])
            error_map = feature_rms(target[key] - source[key])
            arrays[f"feature{index}_pytorch_rms"] = source_map
            arrays[f"feature{index}_jittor_rms"] = target_map
            arrays[f"feature{index}_error_rms"] = error_map
        np.savez_compressed(DATA / "raft_features.npz", **arrays)
        write_iteration_metrics(source, target)
    write_tensor_alignment()
    metadata = {
        "schema_version": 1,
        "source_capture": {
            "logical_path": "data-fs://RAFT/results/dsh_torchport/traces/source.npz",
            "sha256": sha256(source_path),
            "bytes": source_path.stat().st_size,
        },
        "target_capture": {
            "logical_path": "data-fs://RAFT/results/dsh_torchport/traces/target.npz",
            "sha256": sha256(target_path),
            "bytes": target_path.stat().st_size,
        },
        "feature_reduction": "sqrt(mean(channel_value ** 2)) for each spatial location",
        "feature_shape": [1, 256, 48, 64],
        "flow_difference": "pixelwise endpoint difference between PyTorch and Jittor flow at the same iteration",
    }
    (DATA / "raft_features_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
