"""Leave-one-out check and rebuild of the fixed target repeat baseline (judge-only).

All valid controller reference runs are used. For each held-out run, the baseline is
the per-tensor maximum pairwise distance among the other runs; the held-out run is then
judged with the pilot rule err(run, fp64) <= k * max(source noise, baseline) + floor
(`ustat` sum dropped, max_abs and p99). Negative controls are judged with the baseline
from all runs. The all-run baseline is written as a new judge artifact; the old one is
kept unchanged. Diagnostic only; no verify result is rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {n: archive[n] for n in archive.files if n.startswith("output.")}


def stats(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.abs(a - b).ravel()
    return np.asarray([diff.max(), np.quantile(diff, 0.99)]) if diff.size else np.zeros(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--references", type=Path, nargs="+", required=True)
    parser.add_argument("--controls", nargs="+", required=True, help="name=path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-digest", required=True)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--floor", type=float, default=1e-7)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    controls = dict(item.split("=", 1) for item in args.controls)
    source, fp64, alt = (load(args.oracle / f) for f in
                         ("outputs.npz", "train_fp64.npz", "train_alt.npz"))
    refs = [load(p) for p in args.references]
    ctrl = {name: load(Path(p)) for name, p in controls.items()}
    names = [n for n in source]
    for run in [*refs, *ctrl.values()]:
        if set(run) != set(names):
            raise ValueError("输出集合与 oracle 不同")
    n = len(refs)
    pairs = list(itertools.combinations(range(n), 2))
    baseline: dict[str, np.ndarray] = {}
    loo_usage = np.zeros((n, len(names)))
    full_usage = np.zeros((n, len(names)))
    ctrl_usage = {name: np.zeros(len(names)) for name in ctrl}
    for t, name in enumerate(names):
        sub = slice(0, 2) if name.startswith("output.ustat.") else slice(None)
        f = np.asarray(fp64[name][sub], np.float64)
        s = np.asarray(source[name][sub], np.float64)
        noise = np.maximum(stats(s, f), stats(s, np.asarray(alt[name][sub], np.float64)))
        values = [np.asarray(r[name][sub], np.float64) for r in refs]
        pair = {p: stats(values[p[0]], values[p[1]]) for p in pairs}
        diameter = np.max(np.stack(list(pair.values())), axis=0)
        baseline[name] = diameter
        full_basis = np.maximum(noise, diameter)
        for i in range(n):
            others = np.max(np.stack([v for p, v in pair.items() if i not in p]), axis=0)
            basis = np.maximum(noise, others)
            error = stats(values[i], f)
            loo_usage[i, t] = np.max(error / (args.k * basis + args.floor))
            full_usage[i, t] = np.max(error / (args.k * full_basis + args.floor))
        for cname, run in ctrl.items():
            error = stats(np.asarray(run[name][sub], np.float64), f)
            ctrl_usage[cname][t] = np.max(error / (args.k * full_basis + args.floor))
    args.output_dir.mkdir(parents=True)
    out = args.output_dir / "target-repeat-baseline.npz"
    np.savez_compressed(out, **baseline)

    def worst(usage: np.ndarray) -> dict:
        index = int(np.argmax(usage))
        return {"failed_tensors": int(np.sum(usage > 1)), "max_usage": float(usage[index]),
                "worst_tensor": names[index]}

    result = {
        "rule": "drop ustat sum; basis=max(source FP32/FP64/alt noise, reference peer diameter)",
        "k": args.k, "floor": args.floor, "references": len(refs),
        "leave_one_out": [{"run": str(p), **worst(loo_usage[i])}
                          for i, p in enumerate(args.references)],
        "in_sample": [{"run": str(p), **worst(full_usage[i])}
                      for i, p in enumerate(args.references)],
        "controls": {name: worst(u) for name, u in ctrl_usage.items()},
    }
    result["loo_failed_runs"] = sum(r["failed_tensors"] > 0 for r in result["leave_one_out"])
    result["controls_detected"] = sum(c["failed_tensors"] > 0 for c in result["controls"].values())
    record = {
        "profile": "train-repeat-envelope-v1-pilot", "revision": 2,
        "task_digest": args.task_digest,
        "references": [{"path": str(p), "sha256": file_hash(p)} for p in args.references],
        "tensors": len(baseline), "drop_ustat_sum": True,
        "baseline_path": str(out), "baseline_sha256": file_hash(out),
    }
    (args.output_dir / "target-repeat-baseline.json").write_text(json.dumps(record, indent=2), "utf-8")
    (args.output_dir / "loo-check.json").write_text(json.dumps(result, indent=2), "utf-8")
    print(json.dumps({"sha256": record["baseline_sha256"],
                      "loo_failed_runs": result["loo_failed_runs"],
                      "controls_detected": result["controls_detected"]}))
    for row in result["leave_one_out"]:
        print("loo", Path(row["run"]).parent.parent.name, Path(row["run"]).parent.name[-8:],
              row["failed_tensors"], round(row["max_usage"], 3), row["worst_tensor"])
    for cname, row in result["controls"].items():
        print("ctrl", cname, row["failed_tensors"], round(row["max_usage"], 1))


if __name__ == "__main__":
    main()
