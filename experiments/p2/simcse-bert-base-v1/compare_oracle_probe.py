"""Compare the full Harness source oracle with the standalone source probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    args = parser.parse_args()
    probe_report = json.loads((args.probe_dir / "report.json").read_text("utf-8"))
    result = {"losses": {}, "selected_parameters": {}}
    with np.load(args.oracle_dir / "outputs.npz", allow_pickle=False) as oracle:
        with np.load(args.probe_dir / "selected_tensors.npz", allow_pickle=False) as probe:
            for index, step in enumerate(probe_report["step_reports"]):
                key = f"output.loss.{index}"
                observed = np.asarray(oracle[key]).reshape(-1)
                result["losses"][key] = bool(
                    observed.shape == (1,) and float(observed[0]) == step["loss"]
                )
                for name in step["selected_update_norms"]:
                    oracle_key = f"output.param.{index}.{name}"
                    probe_key = f"step_{index}.param.{name}"
                    result["selected_parameters"][oracle_key] = bool(
                        np.array_equal(oracle[oracle_key], probe[probe_key])
                    )
    result["passed"] = all(result["losses"].values()) and all(
        result["selected_parameters"].values()
    )
    (args.oracle_dir / "probe-comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"],
                      "loss_count": len(result["losses"]),
                      "parameter_count": len(result["selected_parameters"])}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
