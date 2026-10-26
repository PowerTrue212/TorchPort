"""Check the frozen source entry against upstream Hugging Face inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from torchport_adapter import TorchPortSimCSE
from transformers import AutoModel, AutoTokenizer

SENTENCES = (
    "A person is playing the guitar.",
    "Someone plays music on a guitar.",
    "The cat is sleeping on the sofa.",
    "A plane lands near the airport.",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint_dir), local_files_only=True)
    upstream = AutoModel.from_pretrained(str(args.checkpoint_dir), local_files_only=True)
    adapter = TorchPortSimCSE()
    loaded = adapter.model.load_state_dict(upstream.state_dict(), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("Source adapter did not load all published weights")
    upstream.eval()
    adapter.eval()
    inputs = tokenizer(SENTENCES, padding=True, truncation=True, max_length=32,
                       return_tensors="pt")
    with torch.no_grad():
        expected = upstream(**inputs, return_dict=True).last_hidden_state[:, 0, :]
        actual = adapter(inputs["input_ids"], inputs["attention_mask"],
                         inputs["token_type_ids"])
    difference = (expected - actual).abs().numpy()
    result = {
        "passed": bool(np.array_equal(expected.numpy(), actual.numpy())),
        "max_abs": float(difference.max()),
        "shape": list(actual.shape),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentences": list(SENTENCES),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
