from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

OUT = Path(__file__).resolve().parent

METHODS = ["V4 Flash + TP", "V4 Pro + TP", "DSH", "Codex"]
ROUNDS = [3, 1, 1, 5]
TOKENS = [0.92, 1.08, 0.95, 0.43]
STATUS = ["pass", "pass", "fail", "pass"]

COLORS = {
    "V4 Flash + TP": "#E89A1E",
    "V4 Pro + TP": "#3F70E0",
    "DSH": "#E3342F",
    "Codex": "#6847B8",
}


def bubble(ax, x: float, y: float, size: float, color: str) -> None:
    for scale, alpha in [(1.24, 0.035), (1.17, 0.055), (1.10, 0.075)]:
        ax.scatter(
            x,
            y,
            s=size * scale,
            color=color,
            edgecolors="none",
            alpha=alpha,
            clip_on=False,
            zorder=2,
        )
    ax.scatter(
        x,
        y,
        s=size,
        color=color,
        edgecolors="none",
        alpha=0.58,
        clip_on=False,
        zorder=3,
    )


def main() -> None:
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
            "ytick.major.size": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = list(range(len(METHODS)))
    max_tokens = max(TOKENS)
    sizes = [(850 + 3050 * token / max_tokens) * 1.2 for token in TOKENS]

    fig, ax = plt.subplots(figsize=(5.9, 3.45), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for xi, method, rounds, tokens, size, status in zip(
        x, METHODS, ROUNDS, TOKENS, sizes, STATUS
    ):
        bubble(ax, xi, rounds, size, COLORS[method])
        label_y = rounds + 1.18 if method == "Codex" else rounds + 1.35
        ax.text(
            xi,
            min(label_y, 6.35),
            method,
            ha="center",
            va="center",
            color="black",
            fontsize=8.8,
            fontweight="bold",
            zorder=7,
        )
        if status == "fail":
            ax.text(
                xi,
                rounds + 0.14,
                f"{tokens:.2f}M",
                ha="center",
                va="center",
                color="white",
                fontsize=9.8,
                fontweight="bold",
                zorder=5,
            )
            ax.text(
                xi,
                rounds - 0.18,
                "FAIL",
                ha="center",
                va="center",
                color="#FF0000",
                fontsize=10.4,
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.8, foreground="white")],
                zorder=6,
            )
        else:
            ax.text(
                xi,
                rounds,
                f"{tokens:.2f}M",
                ha="center",
                va="center",
                color="white",
                fontsize=10.2,
                fontweight="bold",
                zorder=5,
            )

    ax.set_xlim(-0.42, len(METHODS) - 0.58)
    ax.set_ylim(-0.32, 6.65)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("Round", fontsize=11, fontweight="bold")
    ax.set_xlabel("")

    ax.grid(axis="y", color="#d6d6d6", linewidth=0.95, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.tick_params(axis="x", bottom=False, labelbottom=False, length=0)
    ax.tick_params(axis="y", pad=6)
    for label in ax.get_yticklabels():
        label.set_fontsize(10.2)
        label.set_fontweight("bold")

    fig.tight_layout(pad=0.45, rect=(0, 0.075, 1, 1))
    fig.text(
        0.5,
        0.035,
        "Performance across models on the same migration task",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
    )
    fig.savefig(OUT / "plot.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "plot.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
