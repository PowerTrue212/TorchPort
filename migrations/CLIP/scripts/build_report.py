from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, load_config, verify_prepared_files, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the two frozen CLIP result visuals")
    parser.add_argument(
        "--allow-incomplete", action="store_true", help="render missing methods as unavailable"
    )
    return parser.parse_args()


def load_results(config: dict[str, Any], allow_incomplete: bool) -> list[dict[str, Any]]:
    results = []
    for method in config["methods"]:
        path = ROOT / "results" / method["id"] / "result.json"
        if not path.is_file():
            if not allow_incomplete:
                raise FileNotFoundError(
                    f"missing final result for {method['display_name']}: {path}"
                )
            result = {
                "method": method["id"],
                "rn": False,
                "na": False,
                "max_error": None,
                "rounds": 0,
                "failure": {"type": "MissingResult", "message": "not evaluated"},
            }
        else:
            result = json.loads(path.read_text(encoding="utf-8"))
        result["display_name"] = method["display_name"]
        results.append(result)
    return results


def write_tables(results: list[dict[str, Any]], output: Path) -> None:
    fields = ["Method", "RN", "NA", "Max Error", "Rounds"]
    rows = [
        {
            "Method": item["display_name"],
            "RN": str(bool(item["rn"])),
            "NA": str(bool(item["na"])),
            "Max Error": "" if item["max_error"] is None else f"{item['max_error']:.9g}",
            "Rounds": int(item["rounds"]),
        }
        for item in results
    ]
    with (output / "method_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Method | RN | NA | Max Error | Rounds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Method']} | {row['RN']} | {row['NA']} | "
            f"{row['Max Error'] or '—'} | {row['Rounds']} |"
        )
    (output / "method_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_heatmaps(
    config: dict[str, Any], results: list[dict[str, Any]], reference: np.ndarray, output: Path
) -> None:
    matrices: list[np.ndarray | None] = [reference]
    titles = ["PyTorch"]
    for item in results:
        path = ROOT / "results" / item["method"] / "logits.npy"
        matrix = np.load(path, allow_pickle=False) if item["rn"] and path.is_file() else None
        matrices.append(matrix)
        titles.append(item["display_name"])
    available = [matrix for matrix in matrices if matrix is not None]
    low = min(float(matrix.min()) for matrix in available)
    high = max(float(matrix.max()) for matrix in available)
    if low == high:
        high = low + 1.0
    labels = [sample["label"] for sample in config["input"]["samples"]]

    figure, axes = plt.subplots(1, 4, figsize=(15.2, 4.2), constrained_layout=True)
    image = None
    for axis, title, matrix in zip(axes, titles, matrices):
        axis.set_title(title, fontsize=12, weight="bold")
        axis.set_xticks(range(4), labels, rotation=35, ha="right")
        axis.set_yticks(range(4), labels)
        axis.set_xlabel("text")
        if axis is axes[0]:
            axis.set_ylabel("image")
        if matrix is None:
            axis.set_facecolor("#eeeeee")
            axis.text(
                0.5, 0.5, "forward failed", ha="center", va="center", transform=axis.transAxes
            )
            axis.set_xlim(-0.5, 3.5)
            axis.set_ylim(3.5, -0.5)
            continue
        image = axis.imshow(matrix, cmap="viridis", vmin=low, vmax=high)
        threshold = (low + high) / 2.0
        for row in range(4):
            for column in range(4):
                color = "white" if matrix[row, column] < threshold else "black"
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.82, label="CLIP logits")
    figure.savefig(output / "heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_method_table(results: list[dict[str, Any]], output: Path) -> None:
    columns = ["Method", "RN", "NA", "Max Error", "Rounds"]
    cells = []
    for item in results:
        cells.append(
            [
                item["display_name"],
                "Yes" if item["rn"] else "No",
                "Yes" if item["na"] else "No",
                "—" if item["max_error"] is None else f"{item['max_error']:.3e}",
                str(item["rounds"]),
            ]
        )
    figure, axis = plt.subplots(figsize=(9.2, 2.4))
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.30, 0.12, 0.12, 0.25, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.65)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#243B53")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor("#EAF2F8")
    figure.savefig(output / "method_table.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    config = load_config()
    verify_prepared_files()
    results = load_results(config, args.allow_incomplete)
    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    reference = np.load(ROOT / "shared" / "prepared" / "logits_ref.npy", allow_pickle=False)

    write_tables(results, output)
    write_json(
        output / "summary.json",
        {
            "protocol_version": config["protocol_version"],
            "alignment": config["alignment"],
            "results": results,
        },
    )
    plot_heatmaps(config, results, reference, output)
    plot_method_table(results, output)
    print(f"wrote exactly two visuals to {output}: heatmaps.png, method_table.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
