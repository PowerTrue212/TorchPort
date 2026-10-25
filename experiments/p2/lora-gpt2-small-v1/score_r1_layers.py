"""Exploratory Benchmark v1 layer score from frozen r1 trace arrays.

The module list and tolerance used here were selected after the r1 trials.
The result is diagnostic, not a pre-registered benchmark score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def score(checkpoint: Path, source: Path, target: Path) -> dict:
    with np.load(checkpoint) as weights:
        # A weight (or LoRA factor) makes its parent a parameter-owning module.
        # In particular, attn.bias is a causal-mask buffer, not a checkpoint.
        owners = sorted({
            f"model.{name.rsplit('.', 1)[0]}"
            for name in weights.files
            if name.endswith((".weight", ".lora_A", ".lora_B"))
        })

    rows = []
    with np.load(source) as reference, np.load(target) as candidate:
        for name in owners:
            row = {"module": name, "d": None, "score": 0.0}
            if name not in reference or name not in candidate:
                row["reason"] = "missing"
            else:
                ref, got = reference[name], candidate[name]
                if ref.shape != got.shape:
                    row["reason"] = "shape"
                elif not np.isfinite(ref).all() or not np.isfinite(got).all():
                    row["reason"] = "nonfinite"
                else:
                    ref64 = ref.astype(np.float64)
                    got64 = got.astype(np.float64)
                    d = float(np.max(np.abs(got64 - ref64) / (1e-6 + 1e-5 * np.abs(ref64))))
                    row.update(d=d, score=1.0 / max(1.0, d), reason="ok")
            rows.append(row)

    return {
        "status": "exploratory_not_formal",
        "formula": "100/L * sum(1/max(1,d_l))",
        "atol": 1e-6,
        "rtol": 1e-5,
        "checkpoint_count": len(rows),
        "score": 100.0 * sum(row["score"] for row in rows) / len(rows),
        "source": str(source),
        "target": str(target),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score(args.checkpoint, args.source, args.target)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
