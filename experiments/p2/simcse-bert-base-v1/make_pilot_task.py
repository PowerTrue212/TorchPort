"""Write a train-repeat-envelope pilot TaskSpec that points at a frozen baseline file.

The baseline is a judge-only field; the source oracle digest must stay unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from torchport.contracts import load_task  # noqa: E402
from torchport.contracts.identity import file_hash  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-task", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    task = json.loads(args.base_task.read_text("utf-8"))
    task["protocol"].update(
        train_envelope="train-repeat-envelope-v1-pilot",
        train_noise_baseline=str(args.baseline.resolve()),
        train_noise_baseline_sha256=file_hash(args.baseline.resolve()),
    )
    args.output.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", "utf-8")
    before, after = load_task(args.base_task)["digest"], load_task(args.output)["digest"]
    if before != after:
        args.output.unlink()
        raise RuntimeError(f"oracle digest changed: {before} -> {after}")
    print(json.dumps({"task": str(args.output), "digest": after,
                      "baseline_sha256": task["protocol"]["train_noise_baseline_sha256"]}))


if __name__ == "__main__":
    main()
