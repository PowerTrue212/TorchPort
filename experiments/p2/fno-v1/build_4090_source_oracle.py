"""Build source-only FNO oracle from a frozen remote bundle."""
import json
import sys
from pathlib import Path

from torchport.contracts import load_task
from torchport.verification import build_oracle

bundle = Path(sys.argv[1])
task = load_task(bundle / "task.json")
result = build_oracle(task, bundle / "evidence")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["ok"] else 3)
