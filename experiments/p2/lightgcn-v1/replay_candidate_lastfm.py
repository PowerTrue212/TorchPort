"""Replay a frozen Agent candidate on the trained LastFM checkpoint (P_t / B_l inputs).

Uses only the candidate's `load_train_model()` and the module call `model(users, items)`
(pairwise scores). The propagated tables for B_l come from the candidate's own propagation
method (`computer` or `propagate`, recorded in the result). Ranking follows the source:
sigmoid, train positives masked to -1024, Top-20.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


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
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate, source = args.candidate.resolve(), args.source.resolve()
    export, output = args.export.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    import jittor as jt  # noqa: PLC0415

    jt.flags.use_cuda = 1
    os.chdir(candidate)
    sys.path.insert(0, str(candidate))
    import evaluation_entry as entry  # noqa: E402, PLC0415

    model = entry.load_train_model()
    with np.load(export / "weights.npz", allow_pickle=False) as archive:
        weights = {name: archive[name] for name in archive.files}
    own = set(model.state_dict())
    if own != set(weights):
        raise ValueError(f"state names differ: {sorted(own ^ set(weights))}")
    model.load_parameters(weights)
    for name, value in model.state_dict().items():
        if not np.array_equal(np.asarray(value.numpy()), weights[name]):
            raise ValueError(f"checkpoint parameter failed round-trip: {name}")
    train = np.loadtxt(source / "data/lastfm/data1.txt", dtype=np.int64)[:, :2] - 1
    test = np.loadtxt(source / "data/lastfm/test1.txt", dtype=np.int64)[:, :2] - 1
    train_by_user: dict[int, list[int]] = defaultdict(list)
    test_by_user: dict[int, list[int]] = defaultdict(list)
    for user, item in train:
        train_by_user[int(user)].append(int(item))
    for user, item in test:
        test_by_user[int(user)].append(int(item))
    users = np.load(export / "users.npy", allow_pickle=False)
    n_items = weights["embedding_item.weight"].shape[0]
    method = "computer" if hasattr(model, "computer") else "propagate"
    logits, ratings = [], []
    with jt.no_grad():
        user_table, item_table = getattr(model, method)()
        user_table, item_table = user_table.numpy(), item_table.numpy()
        for start in range(0, len(users), 100):
            batch = users[start:start + 100]
            pairs_u = jt.array(np.repeat(batch, n_items).astype(np.int64))
            pairs_i = jt.array(np.tile(np.arange(n_items, dtype=np.int64), len(batch)))
            raw = model(pairs_u, pairs_i)
            scores = jt.sigmoid(raw).numpy().reshape(len(batch), n_items)
            logits.append(raw.numpy().reshape(len(batch), n_items))
            for row, user in enumerate(batch):
                scores[row, train_by_user[int(user)]] = -(1 << 10)
            ratings.append(scores)
    logit_array, rating_array = np.concatenate(logits), np.concatenate(ratings)
    top20 = np.argsort(-rating_array, axis=1)[:, :20]
    source_ratings = np.load(export / "ratings.npy", allow_pickle=False)
    source_top20 = np.load(export / "top20.npy", allow_pickle=False)
    difference = np.abs(rating_array.astype(np.float64) - source_ratings.astype(np.float64))
    result = {"candidate": candidate.name, "table_method": method,
              "loaded_parameters": len(weights), "test_users": int(len(users)),
              "metrics": metrics(top20, users, test_by_user),
              "source_rating_max_abs": float(difference.max()),
              "top20_exact_users": int(np.all(top20 == source_top20, axis=1).sum()),
              "top20_slot_agreement": float(np.mean(top20 == source_top20))}
    output.mkdir(parents=True)
    np.savez(output / "tables.npz", user_table=user_table, item_table=item_table,
             logits=logit_array)
    np.save(output / "ratings.npy", rating_array)
    np.save(output / "top20.npy", top20)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
