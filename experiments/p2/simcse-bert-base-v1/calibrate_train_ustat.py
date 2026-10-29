"""Offline sensitivity analysis for training update statistics.

This script does not change any frozen verification result. It compares two
proposed rules against the original source oracle and target captures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def errors(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).ravel()
    return float(diff.max()), float(np.quantile(diff, 0.99))


def score(error: tuple[float, float], noise: tuple[float, float], k: float, floor: float) -> float:
    return max(error[0] / (k * noise[0] + floor), error[1] / (k * noise[1] + floor))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--floor", type=float, default=1e-7)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    with (
        np.load(args.oracle / "outputs.npz", allow_pickle=False) as source,
        np.load(args.oracle / "train_fp64.npz", allow_pickle=False) as fp64,
        np.load(args.oracle / "train_alt.npz", allow_pickle=False) as alt,
    ):
        targets = [np.load(path, allow_pickle=False) for path in args.targets]
        try:
            result = {
                "evidence": {"oracle": str(args.oracle), "targets": [str(p) for p in args.targets]},
                "rules": {"k": args.k, "floor": args.floor, "sum_sketch_scale": "source sketch L2"},
                "runs": {},
            }
            for path, target in zip(args.targets, targets):
                run = {rule: {"failed": [], "worst": []} for rule in ("original", "drop_sum", "sum_sketch_scale")}
                run["missing"] = sorted(set(source.files) - set(target.files))
                run["alt_equal"] = 0
                run["compared"] = 0
                for name in source.files:
                    if not name.startswith("output."):
                        continue
                    if name not in fp64 or name not in alt or name not in target:
                        continue
                    s, f, a, t = (archive[name] for archive in (source, fp64, alt, target))
                    if not (s.shape == f.shape == a.shape == t.shape):
                        raise ValueError(f"shape mismatch: {name}")
                    run["compared"] += 1
                    run["alt_equal"] += int(np.array_equal(s, a))
                    is_ustat = name.startswith("output.ustat.")
                    for rule in run:
                        if rule in ("missing", "alt_equal", "compared"):
                            continue
                        subset = slice(0, 2) if is_ustat and rule != "original" else slice(None)
                        error = errors(t[subset], f[subset])
                        src_noise = errors(s[subset], f[subset])
                        alt_noise = errors(s[subset], a[subset])
                        noise = (max(src_noise[0], alt_noise[0]), max(src_noise[1], alt_noise[1]))
                        ratio = score(error, noise, args.k, args.floor)
                        if is_ustat and rule == "sum_sketch_scale":
                            update_name = name.replace("output.ustat.", "output.update.", 1)
                            ss, ff, aa, tt = (archive[update_name] for archive in (source, fp64, alt, target))
                            sketch_noise = max(
                                float(np.linalg.norm(np.asarray(ss, np.float64) - np.asarray(ff, np.float64))),
                                float(np.linalg.norm(np.asarray(ss, np.float64) - np.asarray(aa, np.float64))),
                            )
                            sum_noise = max(abs(float(s[2] - f[2])), abs(float(s[2] - a[2])), sketch_noise)
                            sum_error = abs(float(t[2] - f[2]))
                            ratio = max(ratio, sum_error / (args.k * sum_noise + args.floor))
                        row = {"name": name, "ratio": ratio, "margin": 1.0 - ratio}
                        run[rule]["worst"].append(row)
                        if ratio > 1:
                            run[rule]["failed"].append(row)
                for rule in ("original", "drop_sum", "sum_sketch_scale"):
                    data = run[rule]
                    data["worst"] = sorted(data["worst"], key=lambda item: -item["ratio"])[:12]
                    data["failed_count"] = len(data["failed"])
                    data["failed"] = sorted(data["failed"], key=lambda item: -item["ratio"])[:30]
                    data["max_ratio"] = data["worst"][0]["ratio"] if data["worst"] else None
                    data["passed"] = data["failed_count"] == 0 and not run["missing"]
                result["runs"][path.parent.parent.name] = run
        finally:
            for target in targets:
                target.close()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for path, run in result["runs"].items():
        print(path, "compared", run["compared"], "alt_equal", run["alt_equal"])
        for rule in ("original", "drop_sum", "sum_sketch_scale"):
            data = run[rule]
            print(rule, "passed", data["passed"], "failed", data["failed_count"], "max_ratio", data["max_ratio"])


if __name__ == "__main__":
    main()
