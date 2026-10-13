from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from common import ROOT, load_config, method_ids, read_json_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one auditable APT repair round")
    parser.add_argument("--method", required=True, choices=method_ids())
    parser.add_argument("--decision", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--evidence", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = ROOT / "results" / args.method / "rounds.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = read_json_lines(path)
    number = len(prior) + 1
    maximum = int(load_config()["execution"]["max_repair_rounds"])
    if number > maximum:
        raise RuntimeError(f"maximum repair rounds exceeded: {maximum}")
    value = {
        "method": args.method,
        "round": number,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision": args.decision,
        "candidate": args.candidate,
        "evidence": args.evidence,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

