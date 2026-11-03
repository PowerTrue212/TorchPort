"""Exploratory P_t and B_l for LightGCN candidates from the 4090 replay outputs.

P_t: g_j = |m_target - m_source| / tau_j, tau_j = max(3 * |source r1 - source r2|, TAU_FLOOR),
     P = 100/J * sum 1/max(1, g_j). TAU_FLOOR was declared before any candidate was scored.
B_l: checkpoints user_table, item_table, logits (test users x all items, pre-sigmoid).
     fixed: d = max |t - s| / (A + R |s|);  noise: d = max |t - f64| / (K * max |s - f64| + FLOOR).
     B = 100/L * sum 1/max(1, d).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TAU_FLOOR = 1e-3
A, R, K, FLOOR = 1e-6, 1e-5, 8.0, 1e-7
CHECKPOINTS = ("user_table", "item_table", "logits")


def p_t(source: dict, repeat: dict, target: dict) -> dict:
    rows = []
    for name, value in source.items():
        tau = max(3 * abs(value - repeat[name]), TAU_FLOOR)
        gap = abs(target[name] - value)
        rows.append({"metric": name, "source": value, "target": target[name], "tau": tau,
                     "g": gap / tau, "score": 1 / max(1.0, gap / tau)})
    return {"P_t": 100 * sum(row["score"] for row in rows) / len(rows), "rows": rows}


def b_l(tables: dict, run: dict) -> dict:
    rows = []
    for name in CHECKPOINTS:
        s = tables[name].astype(np.float64)
        f = tables[f"{name}_fp64"]
        t = run[name].astype(np.float64)
        if t.shape != s.shape or not np.isfinite(t).all():
            rows.append({"checkpoint": name, "d_fixed": None, "d_noise": None})
            continue
        noise = float(np.abs(s - f).max())
        rows.append({"checkpoint": name, "source_fp64_max_abs": noise,
                     "max_abs_vs_source": float(np.abs(t - s).max()),
                     "d_fixed": float(np.max(np.abs(t - s) / (A + R * np.abs(s)))),
                     "d_noise": float(np.abs(t - f).max() / (K * noise + FLOOR))})

    def score(key: str) -> float:
        return 100 * sum(0.0 if r[key] is None else 1 / max(1.0, r[key]) for r in rows) / len(rows)

    return {"B_fixed": score("d_fixed"), "B_noise": score("d_noise"), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--controls", nargs="*", default=[])
    args = parser.parse_args()
    out = args.out
    read = lambda path: json.loads(path.read_text("utf-8"))  # noqa: E731
    source = read(out / "export-r1/manifest.json")["metrics"]
    repeat = read(out / "export-r2/manifest.json")["metrics"]
    with np.load(out / "source-tables.npz") as archive:
        tables = {name: archive[name] for name in archive.files}
    report = {"status": "exploratory_not_formal", "tau_floor": TAU_FLOOR,
              "source_metrics": source, "source_repeat_metrics": repeat,
              "fixed": {"a": A, "r": R}, "noise": {"k": K, "floor": FLOOR}, "candidates": {}}
    for label in args.candidates:
        runs = []
        for repeat_label in ("a", "b"):
            directory = out / f"{label}-{repeat_label}"
            result = read(directory / "result.json")
            with np.load(directory / "tables.npz") as archive:
                run = {name: archive[name] for name in archive.files}
            runs.append({"replay": result, "P_t": p_t(source, repeat, result["metrics"]),
                         "B_l": b_l(tables, run), "_tables": run})
        repeat_diff = {name: float(np.abs(runs[0]["_tables"][name].astype(np.float64)
                                          - runs[1]["_tables"][name]).max())
                       for name in CHECKPOINTS}
        for run in runs:
            del run["_tables"]
        report["candidates"][label] = {"runs": runs, "target_repeat_max_abs": repeat_diff}
    for label in args.controls:
        result = read(out / label / "result.json")
        report.setdefault("controls", {})[label] = {
            "replay": result, "P_t": p_t(source, repeat, result["metrics"])}
    (out / "score.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    for label, item in report["candidates"].items():
        for run in item["runs"]:
            print(label, "P_t=%.2f" % run["P_t"]["P_t"],
                  "B_fixed=%.2f" % run["B_l"]["B_fixed"], "B_noise=%.2f" % run["B_l"]["B_noise"],
                  "top20_exact=%d" % run["replay"]["top20_exact_users"])
        print(label, "repeat", item["target_repeat_max_abs"])
    for label, item in report.get("controls", {}).items():
        print(label, "P_t=%.2f" % item["P_t"]["P_t"],
              "top20_exact=%d" % item["replay"]["top20_exact_users"])


if __name__ == "__main__":
    main()
