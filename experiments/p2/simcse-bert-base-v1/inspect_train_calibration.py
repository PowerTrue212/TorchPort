"""Check that every observed training output has source calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchport.comparison import compare_noise_calibrated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.oracle_dir
    report = compare_noise_calibrated(
        root / "outputs.npz", root / "outputs.npz", root / "train_fp64.npz",
        root / "train_alt.npz", k=8.0, require_calibrated=True,
    )
    (root / "calibration-self-check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"],
                      "compared_tensors": report["metrics"]["compared_tensors"],
                      "uncalibrated": report["uncalibrated"],
                      "failed_tensors": report["metrics"]["failed_tensors"]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
