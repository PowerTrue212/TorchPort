"""Replay a trained LastFM checkpoint in the controller Jittor LightGCN reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import jittor as jt
import numpy as np


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def metrics(top20: np.ndarray, users: np.ndarray, test: dict) -> dict[str, float]:
    discount = 1 / np.log2(np.arange(2, 22))
    hits = np.asarray([[item in test[int(user)] for item in row]
                       for user, row in zip(users, top20)])
    counts = np.asarray([len(test[int(user)]) for user in users])
    ideal = np.asarray([discount[: min(count, 20)].sum() for count in counts])
    return {"precision@20": float(hits.sum() / (20 * len(users))),
            "recall@20": float(np.mean(hits.sum(axis=1) / counts)),
            "ndcg@20": float(np.mean((hits * discount).sum(axis=1) / ideal))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    jt.flags.use_cuda = 1
    sys.path.insert(0, str(args.reference.resolve()))
    import train_target_reference as reference  # noqa: E402, PLC0415

    reference.LAYERS = args.layers
    model = reference.LightGCNReference()
    with np.load(args.export / "weights.npz", allow_pickle=False) as archive:
        weights = {name: archive[name] for name in archive.files}
    if set(weights) != set(model.state_dict()):
        raise ValueError("checkpoint parameter names differ")
    model.load_parameters(weights)
    for name, value in model.state_dict().items():
        if not np.array_equal(value.numpy(), weights[name]):
            raise ValueError(f"checkpoint parameter failed round-trip: {name}")
    train = np.loadtxt(args.source / "data" / "lastfm" / "data1.txt", dtype=np.int64)[:, :2] - 1
    test_data = np.loadtxt(args.source / "data" / "lastfm" / "test1.txt", dtype=np.int64)[:, :2] - 1
    train_by_user: dict[int, list[int]] = defaultdict(list)
    test_by_user: dict[int, list[int]] = defaultdict(list)
    for user, item in train:
        train_by_user[int(user)].append(int(item))
    for user, item in test_data:
        test_by_user[int(user)].append(int(item))
    users = np.load(args.export / "users.npy", allow_pickle=False)
    user_table, item_table = model.computer()
    ratings = []
    for start in range(0, len(users), 100):
        batch = users[start:start + 100]
        scores = jt.sigmoid(user_table[jt.array(batch)] @ item_table.transpose()).numpy()
        for row, user in enumerate(batch):
            scores[row, train_by_user[int(user)]] = -(1 << 10)
        ratings.append(scores)
    rating_array = np.concatenate(ratings)
    top20 = np.argsort(-rating_array, axis=1)[:, :20]
    source_ratings = np.load(args.export / "ratings.npy", allow_pickle=False)
    source_top20 = np.load(args.export / "top20.npy", allow_pickle=False)
    if rating_array.shape != source_ratings.shape:
        raise ValueError("rating shape differs")
    difference = np.abs(rating_array.astype(np.float64) - source_ratings.astype(np.float64))
    result = {"layers": args.layers, "weights_sha256": sha256(args.export / "weights.npz"),
              "graph_sha256": sha256(args.reference / "graph-r1.npz"),
              "train_data_sha256": sha256(args.source / "data" / "lastfm" / "data1.txt"),
              "test_data_sha256": sha256(args.source / "data" / "lastfm" / "test1.txt"),
              "loaded_parameters": len(weights), "test_users": len(users),
              "metrics": metrics(top20, users, test_by_user),
              "source_rating_max_abs": float(difference.max()),
              "source_rating_p99": float(np.quantile(difference, 0.99)),
              "top20_exact_users": int(np.all(top20 == source_top20, axis=1).sum()),
              "top20_slot_agreement": float(np.mean(top20 == source_top20))}
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "ratings.npy", rating_array)
    np.save(args.output_dir / "top20.npy", top20)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
