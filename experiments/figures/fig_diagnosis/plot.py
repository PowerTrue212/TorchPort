from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
DATA = OUT / "data.csv"

COLORS = {
    "Codex": "#B69AD6",
    "DSH + TP": "#3F70E0",
}


def main() -> None:
    rows = list(csv.DictReader(DATA.open()))
    methods = ["Codex", "DSH + TP"]
    values = {row["method"]: int(row["mismatches"]) for row in rows}
    counts = [values[method] for method in methods]
    y = [0.72, 0.28]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "font.size": 10,
            "font.weight": "normal",
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.25,
            "axes.labelweight": "bold",
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5,
            "ytick.major.size": 0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(5.9, 3.45), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.barh(
        y,
        counts,
        height=0.20,
        color=[COLORS[method] for method in methods],
        edgecolor="none",
        alpha=0.90,
        zorder=3,
    )

    for bar, value in zip(bars, counts):
        ax.text(
            value + 180,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            ha="left",
            fontsize=10.2,
            fontweight="bold",
            color="#222222",
            zorder=5,
        )

    ratio = counts[0] / counts[1]
    ax.text(
        7750,
        0.12,
        f"{ratio:.1f}× fewer",
        va="center",
        ha="left",
        fontsize=11.2,
        fontweight="bold",
        color="#FF0000",
        zorder=5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=10.2, fontweight="bold")
    ax.set_xlim(-20, 12350)
    ax.set_ylim(0.02, 0.96)
    ax.set_xticks(range(0, 12001, 2000))
    ax.set_xlabel("Error count on the same task", fontsize=11, fontweight="bold")

    for xpos in range(2000, 12001, 2000):
        ax.axvline(
            xpos,
            linestyle=":",
            linewidth=0.95,
            color="#b8b8b8",
            alpha=0.75,
            zorder=0,
        )
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#333333")

    ax.tick_params(axis="x", pad=6)
    ax.tick_params(axis="y", pad=6)
    for label in ax.get_xticklabels():
        label.set_fontsize(10.2)
        label.set_fontweight("bold")

    fig.tight_layout(pad=0.45)
    fig.savefig(OUT / "plot.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "plot.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
