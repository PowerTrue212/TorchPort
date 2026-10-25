"""Trusted controller-only finalization after an Agent session ends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    _project = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_project), str(_project / "src")]

from experiments.harness.agents.mcp_server import make_gateway  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--stop-reason", required=True)
    args = parser.parse_args()
    result = make_gateway(args.trial).certify(args.stop_reason)
    print(json.dumps({
        "passed": result["passed"], "valid": result["valid"],
        "success": result["success"], "result": str(Path(result["official_report"]).parent / "result.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
