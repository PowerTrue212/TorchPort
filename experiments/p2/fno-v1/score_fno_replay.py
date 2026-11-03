"""Exploratory P_t and B_l for the FNO candidates from probe_fno_replay.py outputs.

P_t: one metric, mean relative L2 on the 50 Darcy test samples.
     tau = max(3 * |source r1 - source r2|, TAU_FLOOR); P = 100 / max(1, |m_t - m_s| / tau).
     TAU_FLOOR = 1e-3 was declared before any candidate was scored.
B_l: checkpoints lifting, fno_blocks, projection.
     fixed: d = max |t - s| / (A + R |s|);  noise: d = max |t - f64| / (K * max |s - f64| + FLOOR).
Controls reuse the earlier task-metrics replays (reference, output + 0.001) for P_t only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TAU_FLOOR = 1e-3
A, R, K, FLOOR = 1e-6, 1e-5, 8.0, 1e-7
CHECKPOINTS = ("lifting", "fno_blocks", "projection")


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {k: np.asarray(archive[k], dtype=np.float64) for k in archive.files}


def p_t(source: float, repeat: float, value: float) -> dict:
    tau = max(3 * abs(source - repeat), TAU_FLOOR)
    gap = abs(value - source)
    return {"source": source, "target": value, "tau": tau, "g": gap / tau,
            "P_t": 100 / max(1.0, gap / tau)}


def b_l(src: dict, f64: dict, run: dict) -> dict:
    rows = []
    for name in CHECKPOINTS:
        s, f, t = src[name], f64[name], run.get(name)
        if t is None or t.shape != s.shape or not np.isfinite(t).all():
            rows.append({"checkpoint": name, "d_fixed": None, "d_noise": None,
                         "reason": "missing_shape_or_nonfinite"})
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
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    args = parser.parse_args()
    src, rep, f64 = (load(args.out / f"{n}.npz") for n in ("source-r1", "source-r2", "source-fp64"))
    m_s, m_r = float(src["relative_l2"].mean()), float(rep["relative_l2"].mean())
    report = {"status": "exploratory_not_formal", "tau_floor": TAU_FLOOR,
              "source_mean_relative_l2": m_s, "source_repeat_mean_relative_l2": m_r,
              "source_repeat_output_max_abs": float(np.abs(src["output"] - rep["output"]).max()),
              "source_fp64_mean_relative_l2": float(f64["relative_l2"].mean()),
              "fixed": {"a": A, "r": R}, "noise": {"k": K, "floor": FLOOR},
              "candidates": {}, "controls": {}}
    for label in args.candidates:
        runs = [load(args.out / f"{label}-{r}.npz") for r in ("a", "b")]
        report["candidates"][label] = {
            "runs": [{"P_t": p_t(m_s, m_r, float(run["relative_l2"].mean())),
                      "B_l": b_l(src, f64, run),
                      "output_max_abs_vs_source": float(np.abs(run["output"] - src["output"]).max())}
                     for run in runs],
            "target_repeat_max_abs": {name: float(np.abs(runs[0][name] - runs[1][name]).max())
                                      for name in (*CHECKPOINTS, "output")}}
    for label in ("jittor-reference", "jittor-output-bias-001"):
        result = json.loads((args.metrics / label / "result.json").read_text("utf-8"))
        report["controls"][label] = {"P_t": p_t(m_s, m_r, float(result["mean_relative_l2"])),
                                     "max_abs_vs_source": result["max_abs_vs_source"]}
    (args.out / "score.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print("source", m_s, "repeat", m_r, "fp64", report["source_fp64_mean_relative_l2"])
    for label, item in report["candidates"].items():
        for run in item["runs"]:
            rows = " ".join(f"{r['checkpoint']}:df={r['d_fixed']:.3g},dn={r['d_noise']:.3g}"
                            if r["d_fixed"] is not None else f"{r['checkpoint']}:missing"
                            for r in run["B_l"]["rows"])
            print(label, "P_t=%.2f" % run["P_t"]["P_t"], "B_fixed=%.2f" % run["B_l"]["B_fixed"],
                  "B_noise=%.2f" % run["B_l"]["B_noise"],
                  "out_abs=%.3g" % run["output_max_abs_vs_source"], rows)
        print(label, "repeat", item["target_repeat_max_abs"])
    for label, item in report["controls"].items():
        print(label, "P_t=%.2f" % item["P_t"]["P_t"], "max_abs=%.3g" % item["max_abs_vs_source"])


if __name__ == "__main__":
    main()
