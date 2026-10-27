"""Freeze supervised SimCSE NLI triplet batches for the training GATE.

Each step uses a different block of rows, so the contrastive task does not
collapse after one update. Columns follow the official NLI file:
premise (sent0), entailment positive (sent1), contradiction hard negative.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

DATASET_SHA256 = "0747687ec3594fa449d2004fd3757a56c24bf5f7428976fb5b67176775a68d48"
VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
COLUMNS = ("sent0", "sent1", "hard_neg")
STEPS = 3
BATCH = 8
MAX_LENGTH = 32
NAMES = ("input_ids", "attention_mask", "token_type_ids")


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
    # v1 冻结时为 3 步；训练 GATE v2 用 16 步，每步取不同的 8 行
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    steps = args.steps
    if steps < 1:
        raise ValueError("--steps must be positive")
    if sha256(args.dataset) != DATASET_SHA256:
        raise ValueError("NLI file differs from the pinned public dataset")
    if sha256(args.tokenizer_dir / "vocab.txt") != VOCAB_SHA256:
        raise ValueError("BERT vocab differs from the pinned base model")

    rows = []
    with args.dataset.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"Unexpected NLI header: {reader.fieldnames}")
        for index, row in enumerate(reader):
            if index >= steps * BATCH:
                break
            if any(not row[column].strip() for column in COLUMNS):
                raise ValueError(f"Selected NLI row {index} has an empty field")
            rows.append(row)
    if len(rows) != steps * BATCH:
        raise ValueError("NLI file has fewer rows than required")

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), use_fast=True,
                                              local_files_only=True)
    # 固定形状：所有句子补到 MAX_LENGTH；官方 collator 补到 batch 内最长，
    # 被补位置由 attention_mask 屏蔽，CLS 输出只差浮点舍入
    arrays = {name: np.zeros((steps, BATCH, len(COLUMNS), MAX_LENGTH), dtype=np.int64)
              for name in NAMES}
    truncated = 0
    for slot, column in enumerate(COLUMNS):
        texts = [row[column] for row in rows]
        full = tokenizer(texts, truncation=False)["input_ids"]
        truncated += sum(len(ids) > MAX_LENGTH for ids in full)
        encoded = tokenizer(texts, max_length=MAX_LENGTH, truncation=True,
                            padding="max_length", return_tensors="np")
        for name in NAMES:
            value = np.asarray(encoded[name], dtype=np.int64)
            arrays[name][:, :, slot, :] = value.reshape(steps, BATCH, MAX_LENGTH)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, value in arrays.items():
        path = args.output_dir / f"{name}.npy"
        np.save(path, value)
        files[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                       "sha256": sha256(path)}
    manifest = {
        "schema_version": 1,
        "source_dataset_sha256": DATASET_SHA256,
        "tokenizer_vocab_sha256": VOCAB_SHA256,
        "row_indices_zero_based": list(range(steps * BATCH)),
        "step_rule": f"step i uses data rows [{BATCH}*i, {BATCH}*(i+1))",
        "columns": list(COLUMNS),
        "row_sha256": [hashlib.sha256("\t".join(row[c] for c in COLUMNS).encode("utf-8"))
                       .hexdigest() for row in rows],
        "max_length": MAX_LENGTH,
        "padding": "max_length",
        "truncated_sentences": truncated,
        "files": files,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"shape": files["input_ids"]["shape"], "truncated": truncated,
                      "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
