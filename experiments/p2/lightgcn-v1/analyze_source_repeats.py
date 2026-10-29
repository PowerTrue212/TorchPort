"""Read-only source FP32 repeatability check for the LightGCN pilot."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    difference = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).ravel()
    return float(difference.max()), float(np.quantile(difference, 0.99)), int(np.count_nonzero(difference))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fp64", type=Path, required=True)
    parser.add_argument("--alt", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [args.source, args.fp64, args.alt, *args.repeat]
    archives = [np.load(path, allow_pickle=False) for path in paths]
    try:
        source, fp64, alt, *repeats = archives
        names = [name for name in source.files if name.startswith("output.")]
        rows = []
        for name in names:
            values = [source[name], *(archive[name] for archive in repeats)]
            peer = [distance(a, b) for a, b in itertools.combinations(values, 2)]
            source_fp64 = distance(values[0], fp64[name])
            source_alt = distance(values[0], alt[name])
            rows.append({"name": name, "peer_max_abs": max(d[0] for d in peer),
                         "peer_p99": max(d[1] for d in peer),
                         "peer_changed_pairs": sum(d[2] > 0 for d in peer),
                         "source_fp64": source_fp64, "source_alt": source_alt})
    finally:
        for archive in archives:
            archive.close()
    result = {"kind": "pilot-source-repeat-diagnostic", "formal_decision": False,
              "paths": [{"path": str(path), "sha256": digest(path)} for path in paths],
              "runs": 1 + len(args.repeat), "compared_tensors": len(rows),
              "changed_tensors": sum(row["peer_max_abs"] > 0 for row in rows),
              "max_peer_abs": max(row["peer_max_abs"] for row in rows), "rows": rows}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "paths"}}))


if __name__ == "__main__":
    main()
