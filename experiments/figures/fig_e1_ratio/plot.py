from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parent

INK = "#171717"
GRID = "#D7D7D7"
GOLD = "#E8A62A"
CORAL = "#D94B63"


def configure_style() -> None:
    font_path = Path(r"C:\Windows\Fonts\STSONG.TTF")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)

    plt.rcParams.update(
        {
            "font.family": "STSong",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.05,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def save(figure: plt.Figure, stem: str) -> None:
    figure.savefig(ROOT / f"{stem}.png", dpi=360, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(ROOT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


def draw() -> None:
    correct = [
        ("CamoFormer", 0.65, 0.72),
        ("VSCode-v2", 1.10, 1.02),
        ("PlantCamo", 1.97, 1.33),
        ("BiRefNet", 3.00, 1.67),
    ]
    errors = [
        ("eps error", 9.0, 1.31),
        ("Activation error", 62.0, 1.08),
        ("Stronger error", 200.0, 1.22),
        ("Residual error", 100_000.0, 1.55),
    ]

    figure, axis = plt.subplots(figsize=(7.0, 2.75))
    axis.set_xscale("log")
    axis.set_xlim(0.38, 1.5e5)
    axis.set_ylim(0.48, 2.05)

    axis.axvspan(0.38, 2, color="#FFF7D6", alpha=0.62, zorder=0)
    axis.axvspan(2, 4, color="#E9F5EA", alpha=0.78, zorder=0)
    axis.axvspan(4, 1.5e5, color="#FCEBEC", alpha=0.72, zorder=0)
    for threshold in (2, 4):
        axis.axvline(
            threshold,
            color="#575757",
            linestyle=(0, (4, 3)),
            linewidth=1.0,
            zorder=1,
        )

    for name, value, y_value in correct:
        axis.scatter(
            value,
            y_value,
            s=70,
            marker="o",
            color=GOLD,
            edgecolor=INK,
            linewidth=0.85,
            zorder=3,
        )
        axis.annotate(
            f"{name}\n{value:g}",
            (value, y_value),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.8,
            linespacing=1.05,
            color=INK,
        )

    for name, value, y_value in errors:
        axis.scatter(
            value,
            y_value,
            s=82,
            marker="x",
            color=CORAL,
            linewidth=2.2,
            zorder=3,
        )
        alignment = "right" if value >= 1e4 else "left"
        offset = -7 if value >= 1e4 else 7
        display_value = "1e5" if value == 100_000 else f"{value:g}"
        axis.annotate(
            f"{name}\n{display_value}",
            (value, y_value),
            xytext=(offset, 0),
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=8.8,
            linespacing=1.05,
            color=INK,
        )

    axis.text(
        1.05,
        1.98,
        "PASS\n≤ 2",
        ha="center",
        va="top",
        fontsize=8.2,
        linespacing=0.95,
    )
    axis.text(
        2.83,
        1.98,
        "REVIEW\n2–4",
        ha="center",
        va="top",
        fontsize=8.2,
        linespacing=0.95,
    )
    axis.text(
        110,
        1.98,
        "FAIL\n> 4",
        ha="center",
        va="top",
        fontsize=8.2,
        linespacing=0.95,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GOLD,
            markeredgecolor=INK,
            markeredgewidth=0.85,
            markersize=6.5,
            label="Correct candidate",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=CORAL,
            linewidth=0,
            markeredgewidth=2.0,
            markersize=7,
            label="Known error",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        fontsize=8.5,
        handletextpad=0.5,
        borderpad=0.2,
    )

    axis.set_xticks([0.5, 1, 2, 4, 10, 100, 1e5])
    axis.set_xticklabels(["0.5", "1", "2", "4", "10", "100", "1e5"])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_yticks([])
    axis.set_xlabel("Maximum E1 ratio")
    axis.grid(axis="x", which="major", color=GRID, linestyle=(0, (4, 4)), linewidth=0.7)
    axis.set_axisbelow(True)

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(1.05)
    axis.tick_params(
        axis="x",
        which="major",
        direction="in",
        top=True,
        width=0.9,
        length=4.2,
        color=INK,
    )
    axis.tick_params(axis="y", left=False, right=False)

    figure.subplots_adjust(left=0.04, right=0.985, top=0.97, bottom=0.22)
    save(figure, "e1_max_ratio")


if __name__ == "__main__":
    configure_style()
    draw()
