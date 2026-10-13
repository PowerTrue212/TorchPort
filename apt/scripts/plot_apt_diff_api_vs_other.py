#!/usr/bin/env python3
"""Plot the APT human-migration API-vs-other composition."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter


OUTPUT = Path(__file__).resolve().parents[1] / "reports" / "apt_human_diff_api_vs_other.png"
FONT = FontProperties(fname=r"C:\Windows\Fonts\Deng.ttf")
FONT_BOLD = FontProperties(fname=r"C:\Windows\Fonts\Dengb.ttf")
LATIN = FontProperties(family="Arial")


def main() -> None:
    api = [46.4, 17.7]
    other = [53.6, 82.3]
    labels = ["按修改块数量", "按代码改动量"]
    x = [0, 1]

    fig, ax = plt.subplots(figsize=(8.2, 4.9), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    api_color = "#315A7D"
    other_color = "#DDE2E7"
    width = 0.48

    ax.bar(x, api, width=width, color=api_color, edgecolor="white", linewidth=1.5)
    ax.bar(x, other, width=width, bottom=api, color=other_color, edgecolor="white", linewidth=1.5)

    for index in range(2):
        ax.text(
            x[index],
            api[index] / 2,
            f"API  {api[index]:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontproperties=LATIN,
            fontsize=11,
        )
        ax.text(
            x[index],
            api[index] + other[index] / 2,
            f"其他适配  {other[index]:.1f}%",
            ha="center",
            va="center",
            color="#273142",
            fontproperties=FONT,
            fontsize=11,
        )

    ax.set_ylim(0, 103)
    ax.set_xlim(-0.62, 1.62)
    ax.set_xticks(x, labels, fontproperties=FONT, fontsize=11)
    ax.set_yticks([0, 50, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(FONT)
        tick.set_fontsize(10)

    ax.grid(axis="y", color="#E7EAEE", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#AEB6C3")
    ax.tick_params(axis="y", length=0, colors="#687386")
    ax.tick_params(axis="x", length=0, pad=9, colors="#273142")
    ax.set_ylabel("占比（%）", fontproperties=FONT, fontsize=10, color="#4E5968", labelpad=10)

    ax.set_title(
        "APT 人工迁移改动构成",
        fontproperties=FONT_BOLD,
        fontsize=16,
        color="#1D2736",
        pad=18,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
