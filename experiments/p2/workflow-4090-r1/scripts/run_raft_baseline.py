"""Run the leave-one-out repeat baseline for RAFT using the 12 collected references."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/RAFT-Benchmark-v1")
CAL = ROOT / "cal-r1"
ORACLE = ROOT / "oracle-bundle-train-three-r1-r6/evidence/oracle/5586ca6424603914"
LOO = CAL / "scripts/loo_repeat_baseline.py"
PY = "/root/autodl-tmp/envs/source/bin/python"


def main() -> None:
    record = json.loads((CAL / "cal-reports-r1.json").read_text())
    refs = [entry["outputs"] for entry in record["references"]]
    for path in refs:
        if not Path(path).is_file():
            raise SystemExit(f"missing reference outputs: {path}")
    digest = record["references"][0]["task_digest"]
    out_dir = CAL / sys.argv[1] if len(sys.argv) > 1 else CAL / "baseline-r1"
    controls = [f"{entry['name']}={entry['outputs']}" for entry in record["controls"]]
    command = [PY, str(LOO), "--oracle", str(ORACLE), "--references", *refs,
               "--controls", *controls, "--output-dir", str(out_dir),
               "--task-digest", digest, "--k", "8", "--floor", "1e-7"]
    print("running:", " ".join(command), flush=True)
    result = subprocess.run(command, text=True, capture_output=True)
    print(result.stdout)
    print(result.stderr[-4000:])
    print("exit", result.returncode)


if __name__ == "__main__":
    main()
