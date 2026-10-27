"""Exploratory Benchmark v1 layer score for CLIP full-r3 from its trace arrays.

The module list and tolerance were selected after the trial, so the result is
diagnostic, not a pre-registered B_l score. none-r3 called no trace and has no
comparable arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ATOL, RTOL = 1e-6, 1e-5


def owners(weights: Path) -> list[str]:
    """A module owns a checkpoint parameter; top-level parameters belong to `model`."""
    with np.load(weights) as archive:
        names = archive.files
    result = set()
    for name in names:
        parent = name.rsplit(".", 1)[0] if "." in name else ""
        result.add(f"model.{parent}" if parent else "model")
    return sorted(result)


def _key(module: str, files: set[str]) -> str | None:
    # 多输出模块（如 MultiheadAttention）按 .0 记录首个输出
    for key in (module, f"{module}.0"):
        if key in files:
            return key
    return None


def score(weights: Path, source: Path, target: Path) -> dict:
    rows = []
    with np.load(source) as reference, np.load(target) as candidate:
        ref_files, got_files = set(reference.files), set(candidate.files)
        for module in owners(weights):
            row = {"module": module, "d": None, "score": 0.0}
            ref_key, got_key = _key(module, ref_files), _key(module, got_files)
            if ref_key is None or got_key is None:
                row["reason"] = "missing_source" if ref_key is None else "missing_target"
                rows.append(row)
                continue
            ref, got = reference[ref_key], candidate[got_key]
            if ref.shape != got.shape:
                row.update(reason="shape", source_shape=list(ref.shape),
                           target_shape=list(got.shape))
            elif not np.isfinite(ref).all() or not np.isfinite(got).all():
                row["reason"] = "nonfinite"
            else:
                ref64, got64 = ref.astype(np.float64), got.astype(np.float64)
                error = np.abs(got64 - ref64)
                d = float(np.max(error / (ATOL + RTOL * np.abs(ref64))))
                row.update(d=d, score=1.0 / max(1.0, d), reason="ok", key=ref_key,
                           max_abs=float(error.max()), source_absmax=float(np.abs(ref64).max()))
            rows.append(row)
    # 源端没有该模块输出时，模块不可评分，不计入分母
    scored = [row for row in rows if row["reason"] != "missing_source"]
    return {
        "status": "exploratory_not_formal",
        "formula": "100/L * sum(1/max(1,d_l))",
        "atol": ATOL,
        "rtol": RTOL,
        "owner_count": len(rows),
        "checkpoint_count": len(scored),
        "score": 100.0 * sum(row["score"] for row in scored) / len(scored) if scored else None,
        "source": str(source),
        "target": str(target),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.weights, args.source, args.target)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("owner_count", "checkpoint_count", "score")}))


if __name__ == "__main__":
    main()
