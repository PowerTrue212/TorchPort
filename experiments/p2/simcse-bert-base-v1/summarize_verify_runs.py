"""Print every target run in each controller's verify-result.json (not only the first)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

for controller in map(Path, sys.argv[1:]):
    report = json.loads((controller / "verify-result.json").read_text("utf-8"))
    print(controller.name, report.get("failure_code"), json.dumps(report.get("repeats")))
    for run in report.get("results", []):
        metrics = run.get("envelope_metrics") or {}
        print(f"  {run.get('run_id')} {run.get('failure_code')} "
              f"{run.get('runnable')}/{run.get('structural')}/{run.get('numerical')} "
              f"failed={metrics.get('failed_tensors')} max_ratio={metrics.get('max_ratio')}")
        if run.get("failure_code"):
            print("    ", run.get("summary"))
            tail = run.get("stderr_tail") or ""
            if tail:
                print("    stderr:", tail[-600:].replace("\n", " | "))
