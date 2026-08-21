from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
STAGES = ["CV", "RN", "NA", "SA"]
COLORS = {"Official": "#E69F00", "Ours": "#0072B2"}


def load() -> dict[str, list[float]]:
    rows = list(csv.DictReader((ROOT / "data.csv").open(encoding="utf-8")))
    result: dict[str, list[float]] = {}
    for method in COLORS:
        items = {row["stage"]: row["value"] for row in rows if row["method"] == method}
        result[method] = [float(items[stage]) if items[stage] else np.nan for stage in STAGES]
    return result


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    data = load()
    x = np.arange(len(STAGES))
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    for method, values in data.items():
        ax.plot(
            x,
            values,
            color=COLORS[method],
            marker="o",
            markersize=6,
            linewidth=1.8,
            zorder=3,
        )
    ax.scatter(x[2:], [0.5, 0.5], marker="x", s=32, color="#8A8A8A", zorder=3)
    ax.text(2.5, 0.56, "Official: not reached", ha="center", color="#666666", fontsize=8)
    ax.annotate(
        "Official",
        xy=(1, 0),
        xytext=(8, 8),
        textcoords="offset points",
        color=COLORS["Official"],
        fontsize=8,
    )
    ax.annotate(
        "Ours",
        xy=(3, 0),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        color=COLORS["Ours"],
        fontsize=8,
    )
    ax.set_xticks(x, STAGES)
    ax.set_yticks([0, 1], ["Fail", "Pass"])
    ax.set_xlim(-0.15, 3.15)
    ax.set_ylim(-0.12, 1.12)
    ax.set_xlabel("Validation stage")
    ax.set_ylabel("Outcome")
    ax.set_title("SuperRes end-to-end validation")
    ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3)
    fig.text(0.51, -0.01, "CV Convert · RN Run · NA Numerical · SA Strict", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ROOT / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    main()
