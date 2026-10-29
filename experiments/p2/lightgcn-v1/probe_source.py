"""Run a small, fixed LastFM LightGCN source training probe on the V100.

This is preparation evidence, not a frozen TaskSpec or a trained checkpoint.
The official source tree is imported through PYTHONPATH and is left unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.argv = [sys.argv[0], "--dataset=lastfm", "--layer=3", "--recdim=64", "--dropout=0"]

import dataloader  # noqa: E402
import model as upstream_model  # noqa: E402
import world  # noqa: E402


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    torch.manual_seed(2020)
    np.random.seed(2020)
    torch.use_deterministic_algorithms(True)
    data = dataloader.LastFM()
    network = upstream_model.LightGCN(world.config, data).to(world.device)
    users = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=world.device)
    pos = torch.tensor([int(data.allPos[u][0]) for u in range(4)], dtype=torch.long, device=world.device)
    neg = torch.tensor([int(data.allNeg[u][0]) for u in range(4)], dtype=torch.long, device=world.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    started = time.monotonic()
    before = network.getUsersRating(users).detach().cpu().numpy()
    losses = []
    for _ in range(2):
        optimizer.zero_grad()
        bpr, reg = network.bpr_loss(users, pos, neg)
        loss = bpr + 1e-4 * reg
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    after = network.getUsersRating(users).detach().cpu().numpy()
    result = {
        "kind": "source_probe_only",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(world.device),
        "users": data.n_users,
        "items": data.m_items,
        "train_interactions": data.trainDataSize,
        "graph_nnz": int(network.Graph._nnz()),
        "rating_shape": list(before.shape),
        "rating_before_sha256": digest(before),
        "rating_after_sha256": digest(after),
        "losses": losses,
        "wall_seconds": time.monotonic() - started,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
    }
    output = Path(os.environ["TP_LIGHTGCN_PROBE_OUT"])
    raw = output.with_suffix(".npz")
    if output.exists() or raw.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw,
        rating_before=before,
        rating_after=after,
        user_embedding=network.embedding_user.weight.detach().cpu().numpy(),
        item_embedding=network.embedding_item.weight.detach().cpu().numpy(),
    )
    result["raw_npz"] = str(raw)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
