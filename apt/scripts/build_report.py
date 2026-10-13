from __future__ import annotations

import argparse
import csv
import json
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, load_config, verify_reference_files, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the two frozen APT result visuals")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def load_results(config: dict[str, Any], allow_incomplete: bool) -> list[dict[str, Any]]:
    values = []
    for method in config["methods"]:
        path = ROOT / "results" / method["id"] / "result.json"
        if not path.is_file():
            if not allow_incomplete:
                raise FileNotFoundError(path)
            value = {
                "method": method["id"],
                "rn": False,
                "average_accuracy_percent": None,
                "forgetting_percent": None,
                "delta_accuracy_pp": None,
                "rounds": 0,
                "failure": {"type": "MissingResult", "message": "not evaluated"},
            }
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
        value["display_name"] = method["display_name"]
        values.append(value)
    return values


def write_tables(results: list[dict[str, Any]], output) -> None:
    fields = ["Method", "RN", "Avg Acc (%)", "Forgetting (%)", "Delta Acc (pp)", "Rounds"]
    rows = []
    for item in results:
        rows.append(
            {
                "Method": item["display_name"],
                "RN": str(bool(item["rn"])),
                "Avg Acc (%)": ""
                if item["average_accuracy_percent"] is None
                else f"{item['average_accuracy_percent']:.4f}",
                "Forgetting (%)": ""
                if item["forgetting_percent"] is None
                else f"{item['forgetting_percent']:.4f}",
                "Delta Acc (pp)": ""
                if item["delta_accuracy_pp"] is None
                else f"{item['delta_accuracy_pp']:+.4f}",
                "Rounds": int(item["rounds"]),
            }
        )
    with (output / "method_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Method | RN | Avg Acc (%) | Forgetting (%) | Delta Acc (pp) | Rounds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Method']} | {row['RN']} | {row['Avg Acc (%)'] or '—'} | "
            f"{row['Forgetting (%)'] or '—'} | {row['Delta Acc (pp)'] or '—'} | "
            f"{row['Rounds']} |"
        )
    (output / "method_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_accuracy_matrices(
    config: dict[str, Any], results: list[dict[str, Any]], reference: np.ndarray, output
) -> None:
    matrices: list[np.ndarray | None] = [reference]
    titles = ["PyTorch"]
    for item in results:
        path = ROOT / "results" / item["method"] / "accuracy_matrix.npy"
        matrices.append(np.load(path, allow_pickle=False) if item["rn"] and path.is_file() else None)
        titles.append(item["display_name"])

    available = [matrix for matrix in matrices if matrix is not None]
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in available])
    low = max(0.0, 5.0 * np.floor(float(finite.min()) / 5.0))
    high = min(100.0, 5.0 * np.ceil(float(finite.max()) / 5.0))
    if low == high:
        low, high = max(0.0, low - 5.0), min(100.0, high + 5.0)

    figure, axes = plt.subplots(1, 4, figsize=(16.2, 4.4), constrained_layout=True)
    image = None
    labels = [str(index) for index in range(1, 11)]
    for axis, title, matrix in zip(axes, titles, matrices):
        axis.set_title(title, fontsize=12, weight="bold")
        axis.set_xticks(range(10), labels)
        axis.set_yticks(range(10), labels)
        axis.set_xlabel("evaluated task i")
        if axis is axes[0]:
            axis.set_ylabel("after training task t")
        if matrix is None:
            axis.set_facecolor("#eeeeee")
            axis.text(0.5, 0.5, "training failed", ha="center", va="center", transform=axis.transAxes)
            axis.set_xlim(-0.5, 9.5)
            axis.set_ylim(9.5, -0.5)
            continue
        masked = np.ma.masked_invalid(matrix)
        image = axis.imshow(masked, cmap="viridis", vmin=low, vmax=high)
        axis.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.35, alpha=0.6)
        axis.tick_params(which="minor", bottom=False, left=False)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.84, label="accuracy (%)")
    figure.savefig(output / "accuracy_matrices.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_method_table(results: list[dict[str, Any]], output) -> None:
    columns = ["Method", "RN", "Avg Acc", "Forgetting", "Delta Acc", "Rounds"]
    cells = []
    for item in results:
        cells.append(
            [
                item["display_name"],
                "Yes" if item["rn"] else "No",
                "—" if item["average_accuracy_percent"] is None else f"{item['average_accuracy_percent']:.2f}%",
                "—" if item["forgetting_percent"] is None else f"{item['forgetting_percent']:.2f}%",
                "—" if item["delta_accuracy_pp"] is None else f"{item['delta_accuracy_pp']:+.2f} pp",
                str(item["rounds"]),
            ]
        )
    figure, axis = plt.subplots(figsize=(11.2, 2.5))
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.25, 0.09, 0.16, 0.17, 0.18, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.7)
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
    reference_manifest = verify_reference_files()
    results = load_results(config, args.allow_incomplete)
    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    reference = np.load(ROOT / "shared" / "reference" / "accuracy_matrix.npy", allow_pickle=False)

    write_tables(results, output)
    write_json(
        output / "summary.json",
        {
            "protocol_version": config["protocol_version"],
            "reference": reference_manifest["summary"],
            "results": results,
        },
    )
    plot_accuracy_matrices(config, results, reference, output)
    plot_method_table(results, output)
    print(f"wrote exactly two visuals: {output / 'accuracy_matrices.png'}, {output / 'method_table.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
