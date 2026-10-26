"""Convert a published BERT checkpoint to a strict TorchPort weight archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertModel


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != args.expected_sha256:
        raise ValueError("Checkpoint SHA-256 does not match the pinned source")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise TypeError("Checkpoint must be a string-keyed state dictionary")
    model = BertModel(BertConfig.from_json_file(str(args.config)))
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    wrong_shape = sorted(key for key in set(expected) & set(state)
                         if tuple(state[key].shape) != tuple(expected[key].shape))
    if missing or extra or wrong_shape:
        raise ValueError(json.dumps({"missing": missing, "extra": extra,
                                     "wrong_shape": wrong_shape}, indent=2))
    weights = {f"model.{key}": value.detach().cpu().numpy().astype("float32")
               for key, value in state.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **weights)
    report = {
        "checkpoint_sha256": checkpoint_hash,
        "weights_sha256": sha256(args.output),
        "num_tensors": len(weights),
        "total_parameters": sum(int(value.size) for value in weights.values()),
        "weights": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in weights.items()},
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "weights"}))


if __name__ == "__main__":
    main()
