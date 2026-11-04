"""Create the three RAFT negative controls from the frozen controller reference.

Each copy changes exactly one source line; the copy is made into a new directory and an
existing directory is never reused. Run on the remote host.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/RAFT-Benchmark-v1")
REF = ROOT / "controller-reference-r1"
OUT = ROOT / "cal-r1" / "negatives"

EDITS = {
    "neg-gamma": (
        "CLIP, GAMMA, MAX_FLOW = 1.0, 0.8, 400",
        "CLIP, GAMMA, MAX_FLOW = 1.0, 0.85, 400",
    ),
    "neg-wd": (
        "LR, WEIGHT_DECAY, EPS, BETAS = 2e-5, 5e-5, 1e-8, (0.9, 0.999)",
        "LR, WEIGHT_DECAY, EPS, BETAS = 2e-5, 5e-4, 1e-8, (0.9, 0.999)",
    ),
    "neg-gru": (
        "        return (1 - z) * h + z * q",
        "        return z * h + (1 - z) * q",
    ),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    record = {}
    for name, (old, new) in EDITS.items():
        dest = OUT / name
        if dest.exists():
            raise FileExistsError(dest)
        shutil.copytree(REF, dest, ignore=shutil.ignore_patterns("__pycache__"))
        entry = dest / "evaluation_entry.py"
        text = entry.read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise ValueError(f"{name}: expected exactly one match, got {text.count(old)}")
        before = sha(entry)
        entry.write_text(text.replace(old, new), encoding="utf-8")
        record[name] = {
            "path": str(dest),
            "old": old,
            "new": new,
            "sha256_before": before,
            "sha256_after": sha(entry),
        }
        print(name, record[name]["sha256_after"])
    (OUT / "negative-controls.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"reference_sha256": sha(REF / "evaluation_entry.py"),
                      "controls": list(record)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
