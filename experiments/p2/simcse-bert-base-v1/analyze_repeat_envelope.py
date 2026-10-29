"""Diagnostic: compare held-out reference runs with a fixed target repeat baseline.

The target baseline comes only from controller-held reference runs. It is never
computed from the Agent candidate being judged. This script does not modify verify.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).ravel()
    return float(diff.max()), float(np.quantile(diff, 0.99))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument("sensitivity", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--calibration", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout", type=Path, nargs="+", required=True)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--floor", type=float, default=1e-7)
    parser.add_argument("--all-tensors", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prior = json.loads(args.sensitivity.read_text("utf-8"))
    names = sorted({
        row["name"]
        for run in prior["runs"].values()
        for row in run["drop_sum"]["failed"]
    })
    # The sensitivity JSON stores only the first 30 failed rows per run. All loss
    # tensors are added explicitly for the selected-tensor diagnostic.
    names.extend(f"output.loss.{i}" for i in range(16) if f"output.loss.{i}" not in names)
    names = sorted(set(names))
    with (
        np.load(args.oracle / "outputs.npz", allow_pickle=False) as source,
        np.load(args.oracle / "train_fp64.npz", allow_pickle=False) as fp64,
        np.load(args.oracle / "train_alt.npz", allow_pickle=False) as alt,
    ):
        if args.all_tensors:
            names = [name for name in source.files if name.startswith("output.")]
        calibration = [np.load(path, allow_pickle=False) for path in args.calibration]
        holdout = [np.load(path, allow_pickle=False) for path in args.holdout]
        try:
            rows = []
            for name in names:
                subset = slice(0, 2) if name.startswith("output.ustat.") else slice(None)
                s, f, a = (archive[name][subset] for archive in (source, fp64, alt))
                calibration_values = [archive[name][subset] for archive in calibration]
                holdout_values = [archive[name][subset] for archive in holdout]
                source_diffs = [stats(s, f), stats(s, a)]
                peer_diffs = [stats(x, y) for x, y in itertools.combinations(calibration_values, 2)]
                noise = tuple(max(d[i] for d in source_diffs) for i in range(2))
                peer = tuple(max(d[i] for d in peer_diffs) for i in range(2))
                basis = tuple(max(noise[i], peer[i]) for i in range(2))
                run_ratios = []
                for value in [*calibration_values, *holdout_values]:
                    error = stats(value, f)
                    run_ratios.append(max(error[i] / (args.k * basis[i] + args.floor) for i in range(2)))
                rows.append({
                    "name": name,
                    "source_noise": noise,
                    "reference_peer_diameter": peer,
                    "run_ratios": run_ratios,
                    "calibration_max_ratio": max(run_ratios[:len(calibration)]),
                    "holdout_max_ratio": max(run_ratios[len(calibration):]),
                })
        finally:
            for archive in [*calibration, *holdout]:
                archive.close()
    result = {
        "scope": "all output tensors" if args.all_tensors else "selected prior failures plus 16 losses",
        "rule": "drop ustat sum; basis=max(source FP32/FP64/alt noise, fixed reference peer diameter)",
        "k": args.k,
        "floor": args.floor,
        "calibration": [str(path) for path in args.calibration],
        "holdout": [str(path) for path in args.holdout],
        "selected_tensors": len(rows),
        "calibration_failures": sum(row["calibration_max_ratio"] > 1 for row in rows),
        "holdout_failures": sum(row["holdout_max_ratio"] > 1 for row in rows),
        "calibration_max_ratio": max(row["calibration_max_ratio"] for row in rows),
        "holdout_max_ratio": max(row["holdout_max_ratio"] for row in rows),
        "rows": sorted(rows, key=lambda row: -row["holdout_max_ratio"]),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
