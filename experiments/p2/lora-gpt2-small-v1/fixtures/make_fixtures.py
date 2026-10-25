"""Create framework-neutral fixtures for the GPT-2 Small + LoRA paired task."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20261017


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Avoid special tokens and keep two distinct sequences so batch handling is exercised.
    input_ids = rng.integers(0, 50_000, size=(2, 16), dtype=np.int64)
    input_path = HERE / "input_ids.npy"
    np.save(input_path, input_ids, allow_pickle=False)

    state: dict[str, np.ndarray] = {}
    for layer in range(12):
        prefix = f"transformer.h.{layer}.attn.c_attn"
        state[f"{prefix}.lora_A"] = rng.normal(0.0, 0.02, size=(8, 768)).astype(np.float32)
        state[f"{prefix}.lora_B"] = rng.normal(0.0, 0.02, size=(1536, 4)).astype(np.float32)
    state_path = HERE / "lora_state.npz"
    np.savez(state_path, **state)

    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "input": {
            "path": input_path.name,
            "dtype": str(input_ids.dtype),
            "shape": list(input_ids.shape),
            "sha256": sha256_file(input_path),
        },
        "lora_state": {
            "path": state_path.name,
            "format": "npz",
            "dtype": "float32",
            "tensor_count": len(state),
            "total_elements": sum(value.size for value in state.values()),
            "sha256": sha256_file(state_path),
            "key_rule": "12 fused-QKV layers; lora_A (8,768), lora_B (1536,4)",
        },
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

