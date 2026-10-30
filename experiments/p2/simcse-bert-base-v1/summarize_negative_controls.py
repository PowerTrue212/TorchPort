"""Summarize hidden `verify` results of controller negative-control directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("controllers", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for controller in args.controllers:
        report = json.loads((controller / "verify-result.json").read_text("utf-8"))
        run = (report.get("results") or [{}])[0]
        metrics = run.get("envelope_metrics") or {}
        rows.append({
            "controller": controller.name,
            "passed": report.get("passed"),
            "failure_code": report.get("failure_code"),
            "runnable": run.get("runnable"),
            "structural": run.get("structural"),
            "numerical": run.get("numerical"),
            "failed_tensors": metrics.get("failed_tensors"),
            "compared_tensors": metrics.get("compared_tensors"),
            "max_ratio": metrics.get("max_ratio"),
            "first": run.get("summary"),
            "candidate_npz": (run.get("artifacts") or report.get("artifacts") or {}).get("candidate"),
            "physical_cost": report.get("physical_cost"),
        })
    text = json.dumps(rows, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", "utf-8")
    for row in rows:
        print(f"{row['controller']:48s} {row['failure_code']} {row['runnable']}/{row['structural']}/"
              f"{row['numerical']} failed={row['failed_tensors']}/{row['compared_tensors']} "
              f"max_ratio={row['max_ratio']:.1f}")
        print(f"    {row['first']}")
        print(f"    {row['candidate_npz']}")


if __name__ == "__main__":
    main()
