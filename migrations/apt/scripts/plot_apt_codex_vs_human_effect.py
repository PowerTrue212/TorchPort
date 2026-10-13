#!/usr/bin/env python3
"""Plot APT migration effect for Codex and the human migration."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D


OUTPUT = Path(__file__).resolve().parents[1] / "reports" / "apt_codex_vs_human_migration_effect.png"
FONT = FontProperties(fname=r"C:\Windows\Fonts\Deng.ttf")
FONT_BOLD = FontProperties(fname=r"C:\Windows\Fonts\Dengb.ttf")


def main() -> None:
    rows = [
        {
            "name": "Codex",
            "reference": 89.1600,
            "migrated": 89.1600037,
            "reference_std": None,
            "migrated_std": None,
            "delta": 0.0000037,
        },
        {
            "name": "人工迁移",
            "reference": 88.7222,
            "migrated": 87.4444,
            "reference_std": 0.5627,
            "migrated_std": 0.3914,
            "delta": -1.2778,
        },
    ]

    reference_color = "#8C97A5"
    migrated_color = "#315A7D"
    connector_color = "#BCC4CD"

    fig, ax = plt.subplots(figsize=(8.4, 4.7), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.78, bottom=0.25)

    for y, row in zip([1, 0], rows):
        start = row["reference"]
        end = row["migrated"]

        ax.plot([start, end], [y, y], color=connector_color, linewidth=2.2, zorder=1)

        if row["reference_std"] is not None:
            ax.errorbar(
                start,
                y,
                xerr=row["reference_std"],
                fmt="none",
                ecolor=reference_color,
                elinewidth=1.1,
                capsize=3,
                zorder=2,
            )
        if row["migrated_std"] is not None:
            ax.errorbar(
                end,
                y,
                xerr=row["migrated_std"],
                fmt="none",
                ecolor=migrated_color,
                elinewidth=1.1,
                capsize=3,
                zorder=2,
            )

        ax.scatter(
            start,
            y,
            s=95,
            facecolor="white",
            edgecolor=reference_color,
            linewidth=2,
            zorder=3,
        )
        ax.scatter(end, y, s=55, color=migrated_color, zorder=4)

        if row["name"] == "Codex":
            ax.text(
                end + 0.12,
                y,
                "89.16 → 89.16    Δ 0.00 pp",
                va="center",
                color="#263343",
                fontproperties=FONT,
                fontsize=10.5,
            )
        else:
            midpoint = (start + end) / 2
            ax.text(
                midpoint,
                y + 0.19,
                "Δ -1.28 pp",
                ha="center",
                color=migrated_color,
                fontproperties=FONT_BOLD,
                fontsize=11,
            )
            ax.text(
                end - 0.07,
                y - 0.18,
                f"{end:.2f}",
                ha="right",
                color=migrated_color,
                fontproperties=FONT,
                fontsize=9.5,
            )
            ax.text(
                start + 0.07,
                y - 0.18,
                f"{start:.2f}",
                ha="left",
                color=reference_color,
                fontproperties=FONT,
                fontsize=9.5,
            )

    ax.set_xlim(86.5, 90.05)
    ax.set_ylim(-0.48, 1.48)
    ax.set_xticks([87, 88, 89, 90])
    ax.set_yticks([0, 1], ["人工迁移", "Codex"], fontproperties=FONT, fontsize=11)
    ax.set_xlabel("平均准确率（%）", fontproperties=FONT, fontsize=10, color="#4E5968", labelpad=9)

    ax.grid(axis="x", color="#E7EAEE", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#AEB6C3")
    ax.tick_params(axis="y", length=0, colors="#273142")
    ax.tick_params(axis="x", length=0, colors="#687386")

    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=reference_color, markeredgewidth=1.8, markersize=7, label="PyTorch 参考"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=migrated_color, markeredgecolor=migrated_color, markersize=6, label="Jittor 迁移结果"),
    ]
    ax.legend(
        handles=legend,
        loc="upper left",
        frameon=False,
        ncol=2,
        prop=FONT,
        bbox_to_anchor=(0, 1.01),
        handletextpad=0.5,
        columnspacing=1.5,
    )

    ax.set_title("APT 迁移前后准确率对比", fontproperties=FONT_BOLD, fontsize=16, color="#1D2736", pad=38)
    fig.text(
        0.5,
        0.025,
        "人工结果为 3 seeds 均值±标准差；Codex 为冻结单次运行。实验协议不同，仅比较各自迁移前后变化。",
        ha="center",
        color="#687386",
        fontproperties=FONT,
        fontsize=8.5,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
