from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


OUT = Path(__file__).with_name("v3-pilot-summary.png")
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
BOLD_FONT_PATH = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
BOLD = FontProperties(fname=str(BOLD_FONT_PATH)) if BOLD_FONT_PATH.exists() else FONT


runs = [
    {
        "label": "PlantCamo\n诊断 none-r1",
        "wall": 77.742,
        "tokens": 34.692547,
        "cost": 1.47542468,
        "feedback": "diagnostic",
        "integrity": "valid",
        "outcome": "structure",
        "ratio": None,
        "offset": (8, 8),
        "ha": "left",
    },
    {
        "label": "PlantCamo\n定位 full-r1",
        "wall": 31.962,
        "tokens": 9.271686,
        "cost": 0.70483004,
        "feedback": "localized",
        "integrity": "valid",
        "outcome": "pass",
        "ratio": 1.2,
        "offset": (8, 8),
        "ha": "left",
    },
    {
        "label": "VSCode-V2-T\n定位 full-r0",
        "wall": 66.334,
        "tokens": 79.488352,
        "cost": 2.66311016,
        "feedback": "localized",
        "integrity": "valid historical",
        "outcome": "pass",
        "ratio": 1.0736175347,
        "offset": (10, 10),
        "ha": "left",
    },
    {
        "label": "VSCode-V2-T\n诊断 none-r1",
        "wall": 56.626,
        "tokens": 74.328976,
        "cost": 2.49515428,
        "feedback": "diagnostic",
        "integrity": "invalid",
        "outcome": "pass",
        "ratio": 1.2257102236,
        "offset": (-10, -8),
        "ha": "right",
    },
]

colors = {"diagnostic": "#3B82F6", "localized": "#F59E0B"}

fig, (ax_cost, ax_e1) = plt.subplots(
    1,
    2,
    figsize=(13.4, 6.8),
    gridspec_kw={"width_ratios": [1.08, 1.0]},
)
fig.patch.set_facecolor("#F7F8FA")

for ax in (ax_cost, ax_e1):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E5E7EB", linewidth=0.8, alpha=0.8)
    ax.tick_params(colors="#4B5563")

for run in runs:
    color = colors[run["feedback"]]
    invalid = run["integrity"] == "invalid"
    marker = "X" if invalid else "o"
    edge = "#B91C1C" if invalid else "white"
    ax_cost.scatter(
        run["wall"],
        run["tokens"],
        s=240,
        marker=marker,
        color=color,
        edgecolor=edge,
        linewidth=2.0,
        zorder=3,
    )
    suffix = "\n反馈泄漏，剔除" if invalid else ""
    ax_cost.annotate(
        f'{run["label"]}\n¥{run["cost"]:.2f}{suffix}',
        (run["wall"], run["tokens"]),
        xytext=run["offset"],
        textcoords="offset points",
        ha=run["ha"],
        fontsize=9.2,
        color="#7F1D1D" if invalid else "#111827",
        fontproperties=FONT,
    )

ax_cost.set_xlim(24, 87)
ax_cost.set_ylim(0, 88)
ax_cost.set_xlabel("DSH 活跃墙钟（分钟）", fontproperties=FONT, color="#374151")
ax_cost.set_ylabel("总 Token（百万）", fontproperties=FONT, color="#374151")
ax_cost.set_title("资源消耗：直接标注费用", fontproperties=BOLD, fontsize=13, loc="left")
ax_cost.text(
    0.0,
    -0.18,
    "蓝色=diagnostic；橙色=localized。墙钟排除会话结束后的空置时间。",
    transform=ax_cost.transAxes,
    fontsize=8.6,
    color="#6B7280",
    fontproperties=FONT,
)

y_positions = list(range(len(runs)))[::-1]
ax_e1.axvspan(-0.72, 0.0, color="#FEF2F2", zorder=0)
ax_e1.axvline(0, color="#D1D5DB", linewidth=1.1)
ax_e1.axvline(2.0, color="#15803D", linewidth=1.6, linestyle="--")
ax_e1.text(
    2.0,
    len(runs) - 0.25,
    "pass 阈值 2",
    ha="right",
    va="bottom",
    fontsize=8.5,
    color="#166534",
    fontproperties=FONT,
)
ax_e1.text(
    -0.36,
    len(runs) - 0.25,
    "结构门",
    ha="center",
    va="bottom",
    fontsize=8.5,
    color="#991B1B",
    fontproperties=FONT,
)

for y, run in zip(y_positions, runs):
    invalid = run["integrity"] == "invalid"
    color = colors[run["feedback"]]
    if run["outcome"] == "structure":
        ax_e1.scatter(-0.36, y, marker="X", s=180, color="#DC2626", zorder=3)
        ax_e1.text(
            -0.27,
            y,
            "STRUCTURE_MISMATCH\n无可比数值分数",
            va="center",
            fontsize=8.5,
            color="#991B1B",
            fontproperties=FONT,
        )
        continue

    ax_e1.scatter(
        run["ratio"],
        y,
        s=170,
        marker="X" if invalid else "o",
        color=color,
        edgecolor="#B91C1C" if invalid else "white",
        linewidth=2,
        zorder=3,
    )
    label = f'{run["ratio"]:.3f}'
    if invalid:
        label += "  原始通过，但剔除"
    ax_e1.text(
        run["ratio"] + 0.06,
        y,
        label,
        va="center",
        fontsize=8.8,
        color="#991B1B" if invalid else "#111827",
        fontproperties=FONT,
    )

ax_e1.set_yticks(y_positions, [run["label"] for run in runs], fontproperties=FONT)
ax_e1.set_xlim(-0.72, 2.28)
ax_e1.set_ylim(-0.65, len(runs) - 0.15)
ax_e1.set_xlabel("v3 noise-envelope max ratio（越小越好）", fontproperties=FONT, color="#374151")
ax_e1.set_title("E1 结果：先过结构门，再比较数值", fontproperties=BOLD, fontsize=13, loc="left")
ax_e1.text(
    0.0,
    -0.18,
    "结构失败没有画成 0；红色 X 表示实验条件失效，而不是数值失败。",
    transform=ax_e1.transAxes,
    fontsize=8.6,
    color="#6B7280",
    fontproperties=FONT,
)

fig.suptitle(
    "TorchPort v3 试跑：成本、验收与实验完整性",
    x=0.055,
    y=0.985,
    ha="left",
    fontsize=17,
    fontproperties=BOLD,
    color="#111827",
)
fig.text(
    0.055,
    0.925,
    "单次试跑只能说明这一次发生了什么，不能据此估计稳定成功率。",
    ha="left",
    fontsize=10,
    fontproperties=FONT,
    color="#4B5563",
)
fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.19, wspace=0.32)
fig.savefig(OUT, dpi=180, facecolor=fig.get_facecolor())
print(OUT)
