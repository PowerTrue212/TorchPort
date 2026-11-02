"""Freeze trained PyTorch weights and LastFM rankings for target replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def metrics(top20: np.ndarray, users: list[int], test: dict) -> dict[str, float]:
    discount = 1 / np.log2(np.arange(2, 22))
    hits = np.asarray([[item in test[user] for item in row] for user, row in zip(users, top20)])
    counts = np.asarray([len(test[user]) for user in users])
    ideal = np.asarray([discount[: min(count, 20)].sum() for count in counts])
    return {"precision@20": float(hits.sum() / (20 * len(users))),
            "recall@20": float(np.mean(hits.sum(axis=1) / counts)),
            "ndcg@20": float(np.mean((hits * discount).sum(axis=1) / ideal))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, checkpoint = args.source.resolve(), args.checkpoint.resolve()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch  # noqa: PLC0415

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    code = source / "code"
    os.chdir(code)
    sys.path.insert(0, str(code))
    sys.argv = [str(code / "main.py"), "--dataset=lastfm", "--model=lgn", "--layer=3",
                "--recdim=64", "--dropout=0", "--seed=20261102", "--topks=[20]",
                "--testbatch=100", "--tensorboard=0", "--multicore=0"]
    import register  # noqa: E402, PLC0415
    import world  # noqa: E402, PLC0415

    dataset = register.dataset
    model = register.MODELS[world.model_name](world.config, dataset).to(world.device)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    users = list(dataset.testDict.keys())
    ratings, top20 = [], []
    with torch.no_grad():
        for start in range(0, len(users), 100):
            batch = users[start:start + 100]
            scores = model.getUsersRating(torch.tensor(batch, device=world.device))
            rows, items = [], []
            for row, positive in enumerate(dataset.getUserPosItems(batch)):
                rows.extend([row] * len(positive))
                items.extend(positive)
            scores[rows, items] = -(1 << 10)
            top20.append(torch.topk(scores, k=20).indices.cpu().numpy())
            ratings.append(scores.cpu().numpy())
    rating_array = np.concatenate(ratings)
    ranking_array = np.concatenate(top20)
    observed = metrics(ranking_array, users, dataset.testDict)
    official = json.loads(args.official_metrics.read_text("utf-8"))["metrics"]
    for key, official_key in (("precision@20", "precision"), ("recall@20", "recall"),
                              ("ndcg@20", "ndcg")):
        if abs(observed[key] - official[official_key][0]) > 1e-12:
            raise ValueError(f"rankings disagree with official metric: {key}")
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "users.npy", np.asarray(users, dtype=np.int64))
    np.save(args.output_dir / "ratings.npy", rating_array)
    np.save(args.output_dir / "top20.npy", ranking_array)
    np.savez_compressed(args.output_dir / "weights.npz",
                        **{name: value.cpu().numpy() for name, value in weights.items()})
    record = {"checkpoint_sha256": digest(checkpoint), "users": len(users),
              "items": dataset.m_items, "metrics": observed,
              "files": {path.name: digest(path) for path in args.output_dir.glob("*.np*")}}
    (args.output_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", "utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
