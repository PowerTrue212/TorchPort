"""Freeze a small LastFM training sequence and random initial embeddings.

Controller-only pilot materials. They are not a trained recommendation checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DATA_SHA256 = "c82af3228cbbda2f55a39244de09ed93292894f1d0d9f1908ebb57e27598fded"
USERS = 1892
ITEMS = 4489
DIM = 64


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("LastFM train data hash changed")
    interactions = np.loadtxt(args.data, dtype=np.int64, usecols=(0, 1)) - 1
    if int(interactions[:, 0].max()) + 1 != USERS or int(interactions[:, 1].max()) + 1 != ITEMS:
        raise ValueError("LastFM user/item count changed")
    positives = [set() for _ in range(USERS)]
    first = {}
    for user, item in interactions:
        user, item = int(user), int(item)
        positives[user].add(item)
        first.setdefault(user, item)
    user_batch = np.asarray([0, 1, 2, 3], dtype=np.int64)
    pos_batch = np.asarray([first[user] for user in user_batch], dtype=np.int64)
    neg_batch = np.asarray([next(item for item in range(ITEMS)
                                 if item not in positives[user]) for user in user_batch], dtype=np.int64)
    steps = 2
    rng = np.random.default_rng(20261029)
    arrays = {
        "users.npy": np.tile(user_batch, (steps, 1)),
        "positive_items.npy": np.tile(pos_batch, (steps, 1)),
        "negative_items.npy": np.tile(neg_batch, (steps, 1)),
    }
    weights = {
        "embedding_user.weight": rng.normal(0, 0.1, (USERS, DIM)).astype(np.float32),
        "embedding_item.weight": rng.normal(0, 0.1, (ITEMS, DIM)).astype(np.float32),
    }
    args.output.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(args.output / name, array, allow_pickle=False)
    np.savez(args.output / "initial-weights.npz", **weights)
    paths = [*arrays, "initial-weights.npz"]
    manifest = {
        "kind": "lightgcn-lastfm-two-step-source-pilot",
        "formal": False,
        "data_sha256": DATA_SHA256,
        "seed": 20261029,
        "users": user_batch.tolist(),
        "positive_items": pos_batch.tolist(),
        "negative_items": neg_batch.tolist(),
        "steps": steps,
        "embedding_shape": {name: list(array.shape) for name, array in weights.items()},
        "files": {name: sha256(args.output / name) for name in paths},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest_sha256": sha256(args.output / "manifest.json")}))


if __name__ == "__main__":
    main()
