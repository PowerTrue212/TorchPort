"""Collect the RAFT calibration verify reports: target output paths, digests, envelope numbers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/RAFT-Benchmark-v1")
EVIDENCE = ROOT / "evidence"
OUT = ROOT / "cal-r1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    trials = [f"cal-r{i:02d}" for i in range(1, 13)]
    controls = ["neg-gamma", "neg-wd", "neg-gru"]
    record = {"references": [], "controls": []}
    for trial in trials:
        path = EVIDENCE / f"verify-train-{trial}.json"
        report = json.loads(path.read_text())
        result = (report.get("results") or [{}])[0]
        candidate = (result.get("artifacts") or {}).get("candidate")
        entry = {
            "trial": trial,
            "task_digest": report.get("task_digest"),
            "passed": report.get("passed"),
            "failure_code": report.get("failure_code"),
            "envelope_metrics": result.get("envelope_metrics"),
            "mismatch_fp32": (report.get("mismatch") or {}).get("fp32"),
            "outputs": candidate,
            "outputs_sha256": sha(Path(candidate)) if candidate and Path(candidate).is_file() else None,
            "physical": report.get("physical_cost"),
            "report": str(path),
        }
        record["references"].append(entry)
    for name in controls:
        path = EVIDENCE / f"verify-train-{name}.json"
        report = json.loads(path.read_text())
        result = (report.get("results") or [{}])[0]
        candidate = (result.get("artifacts") or {}).get("candidate")
        record["controls"].append({
            "name": name,
            "passed": report.get("passed"),
            "failure_code": report.get("failure_code"),
            "summary": report.get("summary"),
            "envelope_metrics": result.get("envelope_metrics"),
            "mismatch_fp32": (report.get("mismatch") or {}).get("fp32"),
            "outputs": candidate,
            "outputs_sha256": sha(Path(candidate)) if candidate and Path(candidate).is_file() else None,
            "report": str(path),
        })
    (OUT / "cal-reports-r1.json").write_text(json.dumps(record, indent=2) + "\n")
    digests = {entry["task_digest"] for entry in record["references"]}
    print("task_digests", len(digests), digests)
    for entry in record["references"]:
        env = entry["envelope_metrics"] or {}
        print("ref", entry["trial"], entry["failure_code"],
              "failed", env.get("failed_tensors"), "max_ratio", round(env.get("max_ratio") or 0, 3),
              "mism_elem", (entry["mismatch_fp32"] or {}).get("mismatched_elements"))
    for entry in record["controls"]:
        env = entry["envelope_metrics"] or {}
        print("neg", entry["name"], entry["failure_code"], "failed", env.get("failed_tensors"),
              "max_ratio", round(env.get("max_ratio") or 0, 1))


if __name__ == "__main__":
    main()
