"""Run a controller-only diagnostic against a frozen SimCSE delivery."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.harness.remote_executor import RemoteExecutor  # noqa: E402

BENCH = Path("D:/科研/SimCSE-Benchmark-v1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--diagnostic-id", required=True)
    parser.add_argument("--tool", choices=("run", "verify"), default="verify")
    args = parser.parse_args()
    controller = args.controller.resolve()
    task = controller / "trial/delivery/official-task-inference.json"
    if not task.is_file():
        raise FileNotFoundError(task)
    output = controller / "diagnostics" / args.diagnostic_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config = json.loads((controller / "mcp-trial.json").read_text(encoding="utf-8"))
    os.environ["TP_EXTRA_ROOTS"] = str(BENCH.resolve())
    executor = RemoteExecutor(ROOT, output / "remote-raw",
                              Path(config["candidate_root"]), config["remote"])
    result = executor.call(args.tool, task, trial=args.diagnostic_id)
    (output / f"{args.tool}-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": result.get("ok"), "passed": result.get("passed"),
                      "failure": result.get("failure"), "output": str(output)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
