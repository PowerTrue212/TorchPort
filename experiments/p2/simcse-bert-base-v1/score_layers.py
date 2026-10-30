"""Exploratory B_l from probe_layers.py outputs; two formulas reported side by side.

fixed:  d_l = max |t - s| / (a + r |s|),               a=1e-6, r=1e-5
noise:  d_l = max_abs(t, fp64) / (k * max_abs(s, fp64) + floor), k=8, floor=1e-7
B = 100/L * sum 1/max(1, d_l) over the 13 checkpoints (final output reported separately).
Checkpoints and tolerances were chosen after the trial; the score is diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

A, R, K, FLOOR = 1e-6, 1e-5, 8.0, 1e-7
CHECKPOINTS = ["model.embeddings", *(f"model.encoder.layer.{i}" for i in range(12))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fp64", type=Path, required=True)
    parser.add_argument("--targets", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, fp64 = np.load(args.source), np.load(args.fp64)
    targets = [np.load(p) for p in args.targets]
    rows = []
    for name in [*CHECKPOINTS, "output"]:
        s = source[name].astype(np.float64)
        f = fp64[name].astype(np.float64)
        noise = float(np.abs(s - f).max())
        row = {"checkpoint": name, "source_absmax": float(np.abs(s).max()),
               "source_fp64_max_abs": noise, "runs": []}
        for t in targets:
            t = t[name].astype(np.float64)
            if t.shape != s.shape or not np.isfinite(t).all():
                row["runs"].append({"reason": "shape_or_nonfinite", "d_fixed": None, "d_noise": None})
                continue
            err = np.abs(t - s)
            d_fixed = float(np.max(err / (A + R * np.abs(s))))
            d_noise = float(np.abs(t - f).max() / (K * noise + FLOOR))
            row["runs"].append({"max_abs_vs_source": float(err.max()), "d_fixed": d_fixed,
                                "d_noise": d_noise})
        if len(targets) > 1:
            a, b = (t[name].astype(np.float64) for t in targets[:2])
            row["target_repeat_max_abs"] = float(np.abs(a - b).max())
        rows.append(row)

    def score(key, run):
        layer = [r for r in rows if r["checkpoint"] != "output"]
        values = [r["runs"][run][key] for r in layer]
        parts = [0.0 if v is None else 1.0 / max(1.0, v) for v in values]
        return 100.0 * sum(parts) / len(parts)

    result = {
        "status": "exploratory_not_formal",
        "checkpoints": len(CHECKPOINTS),
        "fixed": {"a": A, "r": R}, "noise": {"k": K, "floor": FLOOR},
        "B_fixed": [score("d_fixed", i) for i in range(len(targets))],
        "B_noise": [score("d_noise", i) for i in range(len(targets))],
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(json.dumps({k: result[k] for k in ("B_fixed", "B_noise")}))
    for r in rows:
        runs = " | ".join(f"abs={x.get('max_abs_vs_source', float('nan')):.2e} "
                          f"df={x['d_fixed']:.2f} dn={x['d_noise']:.2f}" for x in r["runs"])
        print(f"  {r['checkpoint']:24s} noise={r['source_fp64_max_abs']:.2e} "
              f"rep={r.get('target_repeat_max_abs', 0):.2e} {runs}")


if __name__ == "__main__":
    main()
