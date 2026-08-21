from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def main() -> None:
    rows = list(csv.DictReader((ROOT / "data.csv").open(encoding="utf-8")))
    rounds = [int(row["round"]) for row in rows]
    stages = [int(row["stage"]) for row in rows]
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
        }
    )
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ax.plot(rounds, stages, color="#0072B2", marker="o", markersize=5.5, linewidth=1.8)
    ax.scatter([8], [3], color="#009E73", s=48, zorder=4)
    ax.annotate(
        "verified",
        xy=(8, 3),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        color="#007A59",
        fontsize=8,
    )
    ax.set_xticks(rounds)
    ax.set_yticks([1, 2, 3, 4], ["CV", "RN", "NA", "SA"])
    ax.set_xlim(0.75, 8.25)
    ax.set_ylim(0.8, 4.2)
    ax.set_xlabel("Repair round")
    ax.set_ylabel("Highest stage reached")
    ax.set_title("DSH repair trajectory (SuperRes)")
    ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3)
    fig.text(0.51, -0.01, "CV Convert · RN Run · NA Numerical · SA Strict", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ROOT / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    main()
