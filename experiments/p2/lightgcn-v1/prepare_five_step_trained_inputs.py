"""Freeze five distinct LastFM BPR minibatches for the trained-checkpoint pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DATA_SHA256 = "c82af3228cbbda2f55a39244de09ed93292894f1d0d9f1908ebb57e27598fded"
USERS = 1892
ITEMS = 4489
SEED = 20261102


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
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
    positives = [set() for _ in range(USERS)]
    for user, item in interactions:
        positives[int(user)].add(int(item))
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(interactions), size=(5, 4), replace=False)
    users = interactions[chosen, 0]
    pos = interactions[chosen, 1]
    neg = np.empty_like(users)
    for step in range(5):
        for row in range(4):
            user = int(users[step, row])
            while True:
                item = int(rng.integers(ITEMS))
                if item not in positives[user]:
                    neg[step, row] = item
                    break
    arrays = {"users.npy": users, "positive_items.npy": pos, "negative_items.npy": neg}
    args.output.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(args.output / name, array, allow_pickle=False)
    manifest = {"kind": "lightgcn-lastfm-five-step-trained-checkpoint-pilot",
                "data_sha256": DATA_SHA256, "seed": SEED, "steps": 5, "batch_size": 4,
                "files": {name: sha256(args.output / name) for name in arrays}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
