"""Noise-calibrated 模块隔离验收规则的证据生成脚本。

重新读取 CamoFormer 模块隔离 trace JSON 和任务级指标 JSON，用 k=8、全模块 gating
（不信任文件里存的 gating/k，自己从 ratio_max/ratio_p99 重新判定）重新计算每个
tensor 的裁决，输出图表和 summary.json。

用法（推荐用带 matplotlib 的解释器）：
    E:\\Anaconda\\python.exe docs/evidence/noise_calibrated/build_evidence.py

不写 src/、scripts/ 等受保护目录；只读 CamoFormer/ 和 compatibility/ 下的既有证据，
只写本目录下的 figs/ 和 summary.json。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / "CamoFormer" / "outputs" / "dsh-web-camoformer" / "probe"
NEGCTL = PROBE / "negctl"
TASK_METRICS = ROOT / "CamoFormer" / "outputs" / "task-metrics"
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figs"

K_DEFAULT = 8.0
ATOL_FLOOR = 1e-7


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"证据文件不存在：{path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def rejudge_tensors(tensors: list[dict[str, Any]], k: float = K_DEFAULT) -> list[dict[str, Any]]:
    """忽略文件里存的 gating/passed/k，按 k=8、全模块 gating 自己重新判定。

    文件里的 `gating` 字段在 3080 Ti 的早期 trace 里只标记叶子模块（135/271），
    V100 的 trace 里标记全部模块（271/271）。本脚本统一按“全部模块都 gating”
    重新计算 ratio 与 passed，避免早期叶子口径漏掉非叶子模块（例如 mask_residual）。
    """
    out = []
    for item in tensors:
        if "ratio_max" not in item or "ratio_p99" not in item:
            # uncalibrated / missing：无法重新判定，原样保留并标记
            out2 = dict(item)
            out2["rejudged_gating"] = True
            out2["rejudged_passed"] = None
            out.append(out2)
            continue
        ratio_max = item["ratio_max"]
        ratio_p99 = item["ratio_p99"]
        noise_max = item.get("noise_max_abs", 0.0)
        noise_p99 = item.get("noise_p99_abs", 0.0)
        cand_max = item.get("candidate_max_abs", ratio_max * max(noise_max, ATOL_FLOOR))
        cand_p99 = item.get("candidate_p99_abs", ratio_p99 * max(noise_p99, ATOL_FLOOR))
        limit_max = k * noise_max + ATOL_FLOOR
        limit_p99 = k * noise_p99 + ATOL_FLOOR
        passed = cand_max <= limit_max and cand_p99 <= limit_p99
        out2 = dict(item)
        out2["rejudged_gating"] = True  # 全模块 gating
        out2["rejudged_k"] = k
        out2["rejudged_passed"] = passed
        out2["rejudged_max_ratio"] = max(ratio_max, ratio_p99)
        out.append(out2)
    return out


def calibrated_tensors(report: dict[str, Any]) -> list[dict[str, Any]]:
    return report["isolated"]["calibrated"]["tensors"]


def rejudged_summary(report: dict[str, Any], k: float = K_DEFAULT) -> dict[str, Any]:
    tensors = rejudge_tensors(calibrated_tensors(report), k=k)
    ratios = [t["rejudged_max_ratio"] for t in tensors if t.get("rejudged_max_ratio") is not None]
    failed = [t for t in tensors if t.get("rejudged_passed") is False]
    return {
        "k": k,
        "all_module_gating": True,
        "n_tensors": len(tensors),
        "n_uncalibrated": sum(1 for t in tensors if t.get("rejudged_passed") is None),
        "max_ratio": max(ratios) if ratios else None,
        "n_failed": len(failed),
        "failed_names": [t["name"] for t in failed],
        "passed_overall": len(failed) == 0 and bool(ratios),
        "tensors": tensors,
    }


# ---------------------------------------------------------------------------
# 数据源清单
# ---------------------------------------------------------------------------

RUNS_3080TI = {
    "correct_random": PROBE / "trace-module-isolation-calibrated-r3.json",
    "correct_encoder_ckpt": PROBE / "trace-encoder-ckpt-r1.json",
    "source_repeat": PROBE / "trace-source-repeat-r1.json",
    "convopt0": PROBE / "trace-convopt0-r1.json",
    "mut_ln_eps": NEGCTL / "trace-ln_eps.json",
    "mut_mask_residual": NEGCTL / "trace-mask_residual.json",
    "mut_gelu_tanh": NEGCTL / "trace-gelu_tanh.json",
    "mut_l2_eps": NEGCTL / "trace-l2_eps.json",
}

RUNS_V100 = {
    "correct_random": PROBE / "trace-v100-baseline-r6.json",
    "mut_ln_eps": NEGCTL / "trace-v100-ln_eps.json",
    "mut_mask_residual": NEGCTL / "trace-v100-mask_residual.json",
    "mut_gelu_tanh": NEGCTL / "trace-v100-gelu_tanh.json",
    "mut_l2_eps": NEGCTL / "trace-v100-l2_eps.json",
}

MUTANTS = ["mut_ln_eps", "mut_mask_residual", "mut_gelu_tanh", "mut_l2_eps"]
EFFECTIVE_MUTANTS = ["mut_ln_eps", "mut_mask_residual", "mut_gelu_tanh"]  # l2_eps 当前输入无效


def gpu_label(run_key: str) -> str:
    return "3080 Ti" if run_key.startswith("3080") else "V100"


# ---------------------------------------------------------------------------
# 图 (a) 每模块 ratio 分布
# ---------------------------------------------------------------------------


def fig_a_distribution(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(13, 6))
    labels = []
    positions = []
    pos = 0
    summary_rows = []
    palette = {
        "correct_random": "#2b8a3e",
        "correct_encoder_ckpt": "#2b8a3e",
        "convopt0": "#1c7ed6",
        "mut_ln_eps": "#e03131",
        "mut_mask_residual": "#e03131",
        "mut_gelu_tanh": "#e03131",
        "mut_l2_eps": "#f08c00",
    }
    for gpu_name, runs in data.items():
        for run_key, report in runs.items():
            summ = rejudged_summary(report)
            ratios = np.array(
                [t["rejudged_max_ratio"] for t in summ["tensors"] if t.get("rejudged_max_ratio") is not None],
                dtype=float,
            )
            ratios_clipped = np.clip(ratios, 1e-3, None)
            jitter = (np.random.default_rng(abs(hash((gpu_name, run_key))) % (2**32)).random(len(ratios_clipped)) - 0.5) * 0.3
            color = palette.get(run_key, "#495057")
            ax.scatter(
                np.full_like(ratios_clipped, pos) + jitter,
                ratios_clipped,
                s=10,
                alpha=0.5,
                color=color,
            )
            ax.scatter([pos], [np.max(ratios_clipped)], marker="^", s=70, color="black", zorder=5)
            labels.append(f"{gpu_name}\n{run_key}")
            positions.append(pos)
            summary_rows.append(
                {
                    "gpu": gpu_name,
                    "run": run_key,
                    "max_ratio_k8_allmodule": summ["max_ratio"],
                    "n_failed_k8_allmodule": summ["n_failed"],
                    "passed_k8_allmodule": summ["passed_overall"],
                }
            )
            pos += 1
    ax.axhline(K_DEFAULT, color="red", linestyle="--", linewidth=1.5, label=f"k={K_DEFAULT}")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("max(ratio_max, ratio_p99)  (log scale)")
    ax.set_title(
        "Per-module tensor max(ratio_max, ratio_p99): re-judged k=8, all-module gating\n"
        "(black triangle = run max; red line = k=8 threshold)"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_a_ratio_distribution.png", dpi=150)
    plt.close(fig)
    return {"rows": summary_rows}


# ---------------------------------------------------------------------------
# 图 (b) 固定容差 vs noise-calibrated 判定矩阵
# ---------------------------------------------------------------------------


def fig_b_verdict_matrix(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    row_labels = []
    cols = ["isolated.strict\n(operator-strict)", "output.fp32-inference\n(whole-net)", "noise-calibrated\n(k=8, re-judged)"]
    matrix = []
    for gpu_name, runs in data.items():
        for run_key, report in runs.items():
            strict_passed = report["isolated"]["passed"]
            output_passed = report["output"]["passed"]
            summ = rejudged_summary(report)
            nc_passed = summ["passed_overall"]
            matrix.append([int(strict_passed), int(output_passed), int(nc_passed)])
            row_labels.append(f"{gpu_name} / {run_key}")
            rows.append(
                {
                    "gpu": gpu_name,
                    "run": run_key,
                    "isolated_operator_strict_passed": strict_passed,
                    "output_fp32_inference_passed": output_passed,
                    "noise_calibrated_k8_passed": nc_passed,
                    "output_mismatched_elements": report["output"]["metrics"]["mismatched_elements"],
                    "output_compared_elements": report["output"]["metrics"]["compared_elements"],
                }
            )
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(row_labels))))
    cmap = matplotlib.colors.ListedColormap(["#e03131", "#2b8a3e"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            txt = "PASS" if matrix[i, j] else "FAIL"
            ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.set_title(
        "Fixed tolerance (operator-strict isolated / fp32-inference whole-net output)\n"
        "vs noise-calibrated (k=8 re-judged) verdict matrix"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_b_verdict_matrix.png", dpi=150)
    plt.close(fig)

    # ln_eps 整网固定容差区分力：3080Ti ln_eps / l2_eps / correct_random 的整网失败元素数
    ln_eps_3080 = data["3080 Ti"]["mut_ln_eps"]["output"]["metrics"]["mismatched_elements"]
    l2_eps_3080 = data["3080 Ti"]["mut_l2_eps"]["output"]["metrics"]["mismatched_elements"]
    correct_3080 = data["3080 Ti"]["correct_random"]["output"]["metrics"]["mismatched_elements"]
    return {
        "rows": rows,
        "ln_eps_vs_correct_wholenet_output_mismatch": {
            "ln_eps_mismatched_elements": ln_eps_3080,
            "l2_eps_mismatched_elements": l2_eps_3080,
            "correct_random_mismatched_elements": correct_3080,
            "note": f"整网 fp32-inference 固定容差下三者失败元素数接近（ln_eps={ln_eps_3080}，"
            f"l2_eps={l2_eps_3080}，correct_random={correct_3080}，out of 131072），"
            "说明固定容差不能可靠区分 ln_eps 这个有效 mutation 和干净基线；"
            "需要用 noise-calibrated 的 ratio（ln_eps≈27 vs correct≤3.9）才能分开。",
        },
    }


# ---------------------------------------------------------------------------
# 图 (c) leaf-only vs all-module gating（mask_residual）
# ---------------------------------------------------------------------------


def fig_c_leaf_vs_allmodule(mask_report: dict[str, Any]) -> dict[str, Any]:
    tensors = calibrated_tensors(mask_report)
    leaf_ratios = []
    nonleaf_ratios = []
    leaf_names_failed = []
    nonleaf_names_failed = []
    for t in tensors:
        if "ratio_max" not in t:
            continue
        ratio = max(t["ratio_max"], t["ratio_p99"])
        is_leaf = bool(t.get("gating"))  # 文件里存的 gating 字段 = 当时的叶子标记
        passed_k8 = ratio <= K_DEFAULT
        if is_leaf:
            leaf_ratios.append(ratio)
            if not passed_k8:
                leaf_names_failed.append(t["name"])
        else:
            nonleaf_ratios.append(ratio)
            if not passed_k8:
                nonleaf_names_failed.append(t["name"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (title, vals) in zip(
        axes, [("Leaf modules only\n(old gating convention)", leaf_ratios), ("All modules\n(leaf + composite)", leaf_ratios + nonleaf_ratios)]
    ):
        vals_clipped = np.clip(np.array(vals, dtype=float), 1e-3, None)
        ax.boxplot([vals_clipped], vert=True, showfliers=True)
        ax.scatter(np.full(len(vals_clipped), 1) + (np.random.default_rng(0).random(len(vals_clipped)) - 0.5) * 0.3, vals_clipped, s=10, alpha=0.4, color="#495057")
        ax.axhline(K_DEFAULT, color="red", linestyle="--", label=f"k={K_DEFAULT}")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks([])
        ax.legend()
    axes[0].set_ylabel("max(ratio_max, ratio_p99)  (log scale)")
    fig.suptitle("mask_residual negative control: leaf-only gating misses composite-module anomalies (3080 Ti)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_c_leaf_vs_allmodule.png", dpi=150)
    plt.close(fig)

    return {
        "leaf_only": {
            "n_tensors": len(leaf_ratios),
            "max_ratio": max(leaf_ratios) if leaf_ratios else None,
            "n_failed_k8": len(leaf_names_failed),
            "failed_names": leaf_names_failed,
        },
        "all_module": {
            "n_tensors": len(leaf_ratios) + len(nonleaf_ratios),
            "max_ratio": max(leaf_ratios + nonleaf_ratios) if (leaf_ratios or nonleaf_ratios) else None,
            "n_failed_k8": len(leaf_names_failed) + len(nonleaf_names_failed),
            "failed_names": leaf_names_failed + nonleaf_names_failed,
        },
        "non_leaf_only": {
            "n_tensors": len(nonleaf_ratios),
            "max_ratio": max(nonleaf_ratios) if nonleaf_ratios else None,
            "failed_names": nonleaf_names_failed,
        },
    }


# ---------------------------------------------------------------------------
# 图 (d) ratio vs 模块位置/深度
# ---------------------------------------------------------------------------


def fig_d_ratio_vs_depth(correct_random: dict, correct_encoder: dict, one_mutant: dict, mutant_name: str) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(13, 6))
    series = [
        ("correct_random (random weights, correct)", correct_random, "#2b8a3e", "o"),
        ("correct_encoder_ckpt (real PVTv2 encoder, correct)", correct_encoder, "#1c7ed6", "s"),
        (f"{mutant_name} (effective negative control)", one_mutant, "#e03131", "x"),
    ]
    out_rows = {}
    for label, report, color, marker in series:
        tensors = calibrated_tensors(report)
        xs, ys = [], []
        for t in tensors:
            if "ratio_max" not in t:
                continue
            xs.append(t["position"])
            ys.append(max(t["ratio_max"], t["ratio_p99"]))
        ax.scatter(xs, np.clip(np.array(ys), 1e-3, None), s=14, alpha=0.6, color=color, marker=marker, label=label)
        out_rows[label] = {"n": len(xs), "max_ratio": max(ys) if ys else None}
    ax.axhline(K_DEFAULT, color="red", linestyle="--", linewidth=1.5, label=f"k={K_DEFAULT}")
    ax.set_yscale("log")
    ax.set_xlabel("module position in trace (approx. execution order / depth)")
    ax.set_ylabel("max(ratio_max, ratio_p99)  (log scale)")
    ax.set_title("Ratio vs module position: two correct-weight runs vs one effective mutant (3080 Ti)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_d_ratio_vs_depth.png", dpi=150)
    plt.close(fig)
    return out_rows


# ---------------------------------------------------------------------------
# 图 (e) 重复性
# ---------------------------------------------------------------------------


def fig_e_repeatability(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for gpu_name, runs in data.items():
        for run_key, report in runs.items():
            isolated = report["isolated"]
            src_rep = isolated.get("source_repeatability")
            tgt_rep = isolated.get("repeatability")
            row = {"gpu": gpu_name, "run": run_key}
            if src_rep:
                row["source_mismatched_elements"] = src_rep["metrics"]["mismatched_elements"]
                row["source_max_abs"] = src_rep["metrics"]["max_absolute_error"]
            if tgt_rep:
                row["target_mismatched_elements"] = tgt_rep["metrics"]["mismatched_elements"]
                row["target_max_abs"] = tgt_rep["metrics"]["max_absolute_error"]
                row["target_compared_elements"] = tgt_rep["metrics"]["compared_elements"]
            if "source_mismatched_elements" in row or "target_mismatched_elements" in row:
                rows.append(row)

    if not rows:
        return {"rows": []}

    labels = [f"{r['gpu']}\n{r['run']}" for r in rows]
    src_counts = [r.get("source_mismatched_elements", 0) for r in rows]
    tgt_counts = [r.get("target_mismatched_elements", 0) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(rows))
    width = 0.35
    axes[0].bar(x - width / 2, src_counts, width, label="source self-repeat mismatch", color="#2b8a3e")
    axes[0].bar(x + width / 2, tgt_counts, width, label="target self-repeat mismatch", color="#e03131")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("mismatched elements under operator-strict tolerance")
    axes[0].set_title("Source vs target self-repeat mismatched element counts")
    axes[0].legend()

    src_max = [r.get("source_max_abs", 0.0) for r in rows]
    tgt_max = [r.get("target_max_abs", 0.0) for r in rows]
    axes[1].bar(x - width / 2, src_max, width, label="source self-repeat max_abs", color="#2b8a3e")
    axes[1].bar(x + width / 2, tgt_max, width, label="target self-repeat max_abs", color="#e03131")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("max absolute difference")
    axes[1].set_title("Source vs target self-repeat max_abs")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_e_repeatability.png", dpi=150)
    plt.close(fig)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# 图 (f) 任务级指标
# ---------------------------------------------------------------------------


def fig_f_task_level() -> dict[str, Any]:
    metrics_torch = load(TASK_METRICS / "metrics-torch.json")
    metrics_jittor = load(TASK_METRICS / "metrics-jittor.json")
    pixel_cmp = load(TASK_METRICS / "torch-vs-jittor-pixel-comparison.json")

    datasets = list(metrics_torch["datasets"].keys())
    metric_names = ["mae", "mean_e", "s_measure", "weighted_f"]
    deltas = {m: [] for m in metric_names}
    for ds in datasets:
        t = metrics_torch["datasets"][ds]
        j = metrics_jittor["datasets"][ds]
        for m in metric_names:
            deltas[m].append(abs(j[m] - t[m]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(datasets))
    width = 0.2
    for i, m in enumerate(metric_names):
        vals = np.clip(np.array(deltas[m]), 1e-12, None)
        axes[0].bar(x + (i - 1.5) * width, vals, width, label=m)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylabel("|jittor - torch|  (log scale)")
    axes[0].set_title("Task-level metrics: Jittor vs PyTorch real-checkpoint inference delta (per dataset)")
    axes[0].legend(fontsize=8)

    all_fracs = []
    for ds in pixel_cmp["datasets"].values():
        for img in ds["per_image"]:
            all_fracs.append(img["fraction_pixels_differing"])
    all_fracs = np.array(all_fracs)
    axes[1].hist(all_fracs, bins=50, color="#1c7ed6")
    axes[1].set_xlabel("per-image fraction_pixels_differing (fraction of pixels above abs-diff threshold)")
    axes[1].set_ylabel("number of images")
    axes[1].set_title(f"Per-image pixel-difference fraction histogram (n={len(all_fracs)} images, 4 datasets pooled)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_f_task_level.png", dpi=150)
    plt.close(fig)

    per_dataset_summary = {
        ds: {
            "fraction_pixels_differing": pixel_cmp["datasets"][ds]["fraction_pixels_differing"],
            "mean_absolute_difference_0_255": pixel_cmp["datasets"][ds]["mean_absolute_difference_0_255"],
            "image_count": pixel_cmp["datasets"][ds]["image_count"],
            "metric_deltas": {m: deltas[m][datasets.index(ds)] for m in metric_names},
        }
        for ds in datasets
    }
    return {
        "datasets": datasets,
        "per_dataset": per_dataset_summary,
        "per_image_fraction_stats": {
            "n_images": len(all_fracs),
            "min": float(all_fracs.min()),
            "max": float(all_fracs.max()),
            "mean": float(all_fracs.mean()),
            "median": float(np.median(all_fracs)),
        },
    }


# ---------------------------------------------------------------------------
# 图 (g) margin plot
# ---------------------------------------------------------------------------


def fig_g_margin(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    correct_ratios = []
    mutant_ratios = []
    for gpu_name, runs in data.items():
        for run_key, report in runs.items():
            summ = rejudged_summary(report)
            if summ["max_ratio"] is None:
                continue
            if run_key.startswith("correct"):
                correct_ratios.append((f"{gpu_name}/{run_key}", summ["max_ratio"]))
            elif run_key in EFFECTIVE_MUTANTS:
                mutant_ratios.append((f"{gpu_name}/{run_key}", summ["max_ratio"]))

    max_correct = max(r for _, r in correct_ratios)
    min_mutant = min(r for _, r in mutant_ratios)

    ks = np.arange(2, 33, 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(max_correct, color="#2b8a3e", linestyle="-", linewidth=2, label=f"max correct ratio = {max_correct:.3f}")
    ax.axhline(min_mutant, color="#e03131", linestyle="-", linewidth=2, label=f"min effective-mutant ratio = {min_mutant:.3f}")
    ax.fill_between(ks, max_correct, min_mutant, color="#ffd43b", alpha=0.3, label="feasible k range (separates correct from effective mutants)")
    for k in (4, 8, 16, 32):
        ax.axvline(k, color="gray", linestyle=":", linewidth=0.8)
        ax.text(k, min_mutant * 0.6, f"k={k}", rotation=90, fontsize=8, va="top", ha="right", color="gray")
    ax.set_xlim(2, 32)
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("ratio (log scale)")
    ax.set_title("Separation margin: max correct ratio per run vs min effective-mutant ratio")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_g_margin.png", dpi=150)
    plt.close(fig)

    return {
        "correct_ratios": correct_ratios,
        "effective_mutant_ratios": mutant_ratios,
        "max_correct_ratio": max_correct,
        "min_effective_mutant_ratio": min_mutant,
        "separation_margin_factor": min_mutant / max_correct if max_correct else None,
        "k_default": K_DEFAULT,
        "k_default_separates": max_correct < K_DEFAULT < min_mutant,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    data: dict[str, dict[str, Any]] = {"3080 Ti": {}, "V100": {}}
    for key, path in RUNS_3080TI.items():
        data["3080 Ti"][key] = load(path)
    for key, path in RUNS_V100.items():
        data["V100"][key] = load(path)

    summary: dict[str, Any] = {
        "rule": {
            "source": "src/torchport/comparison/envelope.py::compare_noise_calibrated",
            "definition": (
                "per-module tensor: noise = max(err(source,fp64), err(source,alt)); "
                "pass iff err(target,fp64) <= k*noise + atol_floor for BOTH max_abs and p99; "
                "ALL modules gate (not leaf-only); default k=8, atol_floor=1e-7."
            ),
            "k_used_in_this_report": K_DEFAULT,
            "note": "本报告的全部数字都用 k=8、全模块 gating 重新计算，不信任源 JSON 里存的 k（多数文件存的是探索期 k=4）"
            "或 gating（早期 3080 Ti 文件的 gating 字段只标记叶子模块）。",
        },
        "data_sources": {
            "3080 Ti": {k: str(v.relative_to(ROOT)) for k, v in RUNS_3080TI.items()},
            "V100": {k: str(v.relative_to(ROOT)) for k, v in RUNS_V100.items()},
        },
    }

    print("[1/7] fig_a: ratio 分布 ...")
    summary["fig_a_ratio_distribution"] = fig_a_distribution(data)

    print("[2/7] fig_b: 判定矩阵 ...")
    summary["fig_b_verdict_matrix"] = fig_b_verdict_matrix(data)

    print("[3/7] fig_c: leaf-only vs all-module (mask_residual) ...")
    summary["fig_c_leaf_vs_allmodule"] = fig_c_leaf_vs_allmodule(data["3080 Ti"]["mut_mask_residual"])

    print("[4/7] fig_d: ratio vs depth ...")
    summary["fig_d_ratio_vs_depth"] = fig_d_ratio_vs_depth(
        data["3080 Ti"]["correct_random"],
        data["3080 Ti"]["correct_encoder_ckpt"],
        data["3080 Ti"]["mut_ln_eps"],
        "mut_ln_eps",
    )

    print("[5/7] fig_e: repeatability ...")
    summary["fig_e_repeatability"] = fig_e_repeatability(data)

    print("[6/7] fig_f: task-level ...")
    summary["fig_f_task_level"] = fig_f_task_level()

    print("[7/7] fig_g: margin plot ...")
    summary["fig_g_margin"] = fig_g_margin(data)

    # 附：每个 run 的重判摘要表（供 README 表格直接引用）
    rejudged_table = []
    for gpu_name, runs in data.items():
        for run_key, report in runs.items():
            summ = rejudged_summary(report)
            rejudged_table.append(
                {
                    "gpu": gpu_name,
                    "run": run_key,
                    "max_ratio_k8_allmodule": summ["max_ratio"],
                    "n_failed_k8_allmodule": summ["n_failed"],
                    "n_tensors": summ["n_tensors"],
                    "passed_k8_allmodule": summ["passed_overall"],
                    "failed_names_sample": summ["failed_names"][:5],
                    "stored_k_in_file": report["isolated"]["calibrated"]["k"],
                    "stored_gating_tensors_in_file": report["isolated"]["calibrated"]["metrics"]["gating_tensors"],
                    "stored_total_tensors_in_file": report["isolated"]["calibrated"]["metrics"]["compared_tensors"],
                    "wholenet_output_passed_fp32_inference": report["output"]["passed"],
                    "wholenet_output_mismatched_elements": report["output"]["metrics"]["mismatched_elements"],
                    "wholenet_output_compared_elements": report["output"]["metrics"]["compared_elements"],
                    "isolated_operator_strict_passed": report["isolated"]["passed"],
                    "target_framework": report["target_run"]["capture"]["framework"],
                }
            )
    summary["rejudged_table"] = rejudged_table

    out_path = OUT_DIR / "summary.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=False)
    print(f"写入 {out_path}")
    print(f"图表目录 {FIG_DIR}")


if __name__ == "__main__":
    main()
