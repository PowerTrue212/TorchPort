"""Freeze one Wiki1M SimCSE pair batch using the official tokenizer pattern."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

DATASET_SHA256 = "7b1825863a99aa76479b0456f7c210539dfaeeb69598b41fb4de4f524dd5a706"
VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
ROWS = tuple(range(8))
MAX_LENGTH = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.dataset) != DATASET_SHA256:
        raise ValueError("Wiki1M file differs from the pinned public dataset")
    if sha256(args.tokenizer_dir / "vocab.txt") != VOCAB_SHA256:
        raise ValueError("BERT vocab differs from the pinned base model")

    sentences = []
    with args.dataset.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index > ROWS[-1]:
                break
            if index in ROWS:
                sentence = line.rstrip("\r\n")
                if not sentence:
                    raise ValueError(f"Selected Wiki1M row {index} is empty")
                sentences.append(sentence)
    if len(sentences) != len(ROWS):
        raise ValueError("Wiki1M file has fewer rows than required")

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), use_fast=True,
                                              local_files_only=True)
    encoded = tokenizer(sentences, max_length=MAX_LENGTH, truncation=True,
                        padding="longest", return_tensors="np")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in ("input_ids", "attention_mask", "token_type_ids"):
        value = np.asarray(encoded[name], dtype=np.int64)
        if value.ndim != 2 or value.shape[0] != len(ROWS) or value.shape[1] > MAX_LENGTH:
            raise ValueError(f"Unexpected token shape for {name}: {value.shape}")
        pair = np.stack((value, value), axis=1)
        path = args.output_dir / f"{name}.npy"
        np.save(path, pair)
        files[name] = {"shape": list(pair.shape), "dtype": str(pair.dtype),
                       "sha256": sha256(path)}
    sentence_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest()
                       for text in sentences]
    manifest = {
        "schema_version": 1,
        "source_dataset_sha256": DATASET_SHA256,
        "tokenizer_vocab_sha256": VOCAB_SHA256,
        "row_indices_zero_based": list(ROWS),
        "sentence_sha256": sentence_hashes,
        "pair_rule": "two identical tokenized views per sentence",
        "max_length": MAX_LENGTH,
        "padding": "longest in this fixed batch",
        "files": files,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(ROWS), "shape": files["input_ids"]["shape"],
                      "output_dir": str(args.output_dir)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
