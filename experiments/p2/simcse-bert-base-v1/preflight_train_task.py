"""Run the training task against the untouched seed to build its source oracle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.harness.remote_executor import RemoteExecutor  # noqa: E402
from torchport.runtime import write_json  # noqa: E402

BENCH = Path("D:/科研/SimCSE-Benchmark-v1")
SEED = BENCH / "seed-train/SimCSE"
REMOTE_CONFIG = BENCH / "controller/simcse-none-r0/mcp-trial.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--candidate-root", type=Path, default=SEED)
    args = parser.parse_args()
    controller = args.controller.resolve()
    if controller.exists():
        raise FileExistsError(controller)
    config = json.loads(REMOTE_CONFIG.read_text("utf-8"))
    os.environ["TP_EXTRA_ROOTS"] = str(BENCH)
    controller.mkdir(parents=True)
    executor = RemoteExecutor(ROOT, controller / "remote-raw",
                              args.candidate_root.resolve(), config["remote"])
    report = executor.call("verify", args.task.resolve(), trial=args.trial)
    write_json(controller / "verify-result.json", report)
    print(json.dumps({
        "controller": str(controller),
        "passed": report.get("passed"),
        "failure_code": report.get("failure_code"),
        "oracle_path": report.get("oracle"),
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
