"""Print a compact summary of one or more TorchPort verify JSON reports.

Usage: python summarize_verify.py /path/a.json [/path/b.json ...]
"""
import json
import sys


def short(value, limit=900):
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "..."


for path in sys.argv[1:]:
    try:
        report = json.load(open(path))
    except Exception as exc:  # noqa: BLE001
        print(path, "LOAD_FAILED", exc)
        continue
    result = (report.get("results") or [{}])[0]
    print("==", path)
    print("passed:", report.get("passed"), "| failure_code:", report.get("failure_code"))
    print("summary:", short(report.get("summary"), 400))
    print("repeats:", short(report.get("repeats"), 300))
    print("mismatch:", short(report.get("mismatch"), 300))
    print("envelope:", short(result.get("envelope_metrics"), 1200))
    print("weights:", short((result.get("evidence") or {}).get("weights"), 300))
    print("physical:", short(report.get("physical_cost"), 400))
    print("logical:", short(report.get("logical_cost"), 300))
