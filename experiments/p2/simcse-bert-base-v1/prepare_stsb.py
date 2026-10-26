"""Freeze STS-B test tokens for a published unsupervised SimCSE checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--smoke-pairs", type=int, default=32)
    args = parser.parse_args()

    first, second, labels = [], [], []
    with args.test_csv.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = line.strip().split("\t")
            if len(row) < 7:
                raise ValueError("STS-B row has fewer than seven columns")
            labels.append(float(row[4]))
            first.append(" ".join(row[5].split()))
            second.append(" ".join(row[6].split()))
    if not labels or not np.isfinite(labels).all():
        raise ValueError("STS-B test labels are empty or non-finite")
    if not 0 < args.smoke_pairs <= len(labels):
        raise ValueError("Invalid smoke pair count")

    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint_dir),
                                              local_files_only=True)
    sentences = first + second
    lengths = [len(ids) for ids in tokenizer(sentences, padding=False,
                                             truncation=False)["input_ids"]]
    if max(lengths) > args.max_length:
        raise ValueError(f"Longest sentence has {max(lengths)} tokens; "
                         f"max-length is {args.max_length}")
    encoded = tokenizer(sentences, padding="max_length", truncation=False,
                        max_length=args.max_length, return_tensors="np")
    arrays = {key: np.asarray(encoded[key], dtype=np.int64)
              for key in ("input_ids", "attention_mask", "token_type_ids")}
    arrays["labels"] = np.asarray(labels, dtype=np.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, value in arrays.items():
        path = args.output_dir / f"{key}.npy"
        np.save(path, value)
        files[key] = {"path": str(path), "shape": list(value.shape),
                      "dtype": str(value.dtype), "sha256": sha256(path)}
    smoke_idx = list(range(args.smoke_pairs)) + [
        len(labels) + index for index in range(args.smoke_pairs)
    ]
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        path = args.output_dir / f"smoke_{key}.npy"
        value = arrays[key][smoke_idx]
        np.save(path, value)
        files[f"smoke_{key}"] = {"path": str(path), "shape": list(value.shape),
                                 "dtype": str(value.dtype), "sha256": sha256(path)}
    manifest = {
        "dataset_sha256": sha256(args.test_csv),
        "tokenizer_config_sha256": sha256(args.checkpoint_dir / "tokenizer_config.json"),
        "vocab_sha256": sha256(args.checkpoint_dir / "vocab.txt"),
        "rows": len(labels),
        "sentence_order": "all first sentences, then all second sentences",
        "max_observed_tokens": max(lengths),
        "max_length": args.max_length,
        "smoke_pairs": args.smoke_pairs,
        "files": files,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}))


if __name__ == "__main__":
    main()
