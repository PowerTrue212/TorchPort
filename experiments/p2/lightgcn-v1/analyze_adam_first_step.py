"""Compare captured first-step updates with PyTorch/Jittor Adam formulas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from torchport.capture.training import update_sketch


def predict(initial: np.ndarray, grad: np.ndarray, *, jittor_eps: bool) -> np.ndarray:
    effective_eps = 1e-8 / np.sqrt(1e-3) if jittor_eps else 1e-8
    delta = -0.001 * grad.astype(np.float64) / (np.abs(grad.astype(np.float64)) + effective_eps)
    return (initial.astype(np.float64) + delta).astype(np.float32).astype(np.float64) - initial


def stats(prediction: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    sketch = update_sketch(prediction, 4096)
    difference = np.abs(sketch - observed)
    return {"predicted_update_l2": float(np.linalg.norm(prediction)),
            "sketch_max_abs_error": float(difference.max()),
            "sketch_l2_error": float(np.linalg.norm(difference))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source-grad", type=Path, required=True)
    parser.add_argument("--target-grad", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--target-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    archives = [np.load(path, allow_pickle=False) for path in (
        args.weights, args.source_grad, args.target_grad, args.source_output, args.target_output
    )]
    try:
        weights, source_grad, target_grad, source_output, target_output = archives
        rows = []
        for name in ("embedding_item.weight", "embedding_user.weight"):
            initial = weights[name]
            source_prediction = predict(initial, source_grad[name], jittor_eps=False)
            target_prediction = predict(initial, target_grad[name], jittor_eps=True)
            target_with_torch_eps = predict(initial, target_grad[name], jittor_eps=False)
            source_observed = source_output[f"output.update.0.{name}"]
            target_observed = target_output[f"output.update.0.{name}"]
            rows.append({
                "name": name,
                "source_torch_formula": stats(source_prediction, source_observed),
                "target_jittor_formula": stats(target_prediction, target_observed),
                "target_torch_formula": stats(target_with_torch_eps, target_observed),
                "source_observed_update_l2": float(source_output[f"output.ustat.0.{name}"][0]),
                "target_observed_update_l2": float(target_output[f"output.ustat.0.{name}"][0]),
            })
    finally:
        for archive in archives:
            archive.close()
    result = {"kind": "pilot-adam-first-step-diagnostic", "rows": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
