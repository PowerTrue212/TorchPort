"""Aggregate raw reduction outputs from independent probe processes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SEED = 20261011
SHAPE = (1, 64, 176, 176)


def metrics(values: list[np.ndarray], exact: np.ndarray) -> dict[str, object]:
    flat = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    exact_flat = exact.astype(np.float64).reshape(-1)
    scale = float(np.linalg.norm(exact_flat))
    pairs = [
        left.astype(np.float64) - right.astype(np.float64)
        for i, left in enumerate(flat)
        for right in flat[i + 1 :]
    ]
    mismatches = [
        int(np.count_nonzero(left.view(np.uint32) != right.view(np.uint32)))
        for i, left in enumerate(flat)
        for right in flat[i + 1 :]
    ]
    return {
        "bitwise_equal": max(mismatches) == 0,
        "max_mismatched_elements": max(mismatches),
        "max_abs_run_diff": max(float(np.max(np.abs(delta))) for delta in pairs),
        "max_relative_l2_run_diff": max(float(np.linalg.norm(delta) / scale) for delta in pairs),
    }


def main() -> None:
    trial_dirs = sorted((ROOT / "trials").glob("trial-*"))
    archives = [np.load(path / "outputs.npz") for path in trial_dirs]
    method_names = sorted({key.split("/", 1)[0] for archive in archives for key in archive.files})
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(SHAPE).astype(np.float32)
    exact_sum = x.astype(np.float64).sum(axis=(2, 3))
    exact_avg = x.astype(np.float64).mean(axis=(2, 3), keepdims=True)
    exact = {name: exact_avg if name == "builtin_avgpool" else exact_sum for name in method_names}

    aggregate = {}
    first_runs = {}
    for name in method_names:
        all_values = [archive[key] for archive in archives for key in archive.files if key.startswith(f"{name}/")]
        first_values = [archive[f"{name}/run_00"] for archive in archives]
        aggregate[name] = metrics(all_values, exact[name])
        first_runs[name] = metrics(first_values, exact[name])

    result = {
        "schema_version": 1,
        "trial_count": len(trial_dirs),
        "runs_per_trial": 30,
        "aggregate_90_runs": aggregate,
        "first_run_across_processes": first_runs,
        "trial_result_sha256": {
            path.name: hashlib.sha256((path / "results.json").read_bytes()).hexdigest()
            for path in trial_dirs
        },
    }
    (ROOT / "cross_process.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
