"""Controller adapter around the authors' LightGCN LastFM model.

The only numerical adaptation is casting the graph with embeddings for the
FP64 source calibration branch. The FP32 path uses the original graph and BPR.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
if os.environ.get("TP_LIGHTGCN_DEPS"):
    sys.path.insert(0, os.environ["TP_LIGHTGCN_DEPS"])
sys.path.insert(0, str(SOURCE_ROOT / "code"))
sys.argv = [sys.argv[0], "--dataset=lastfm", "--layer=3", "--recdim=64", "--dropout=0"]

import dataloader  # noqa: E402
import model as upstream_model  # noqa: E402
import torch  # noqa: E402
import world  # noqa: E402


class SourceLightGCN(upstream_model.LightGCN):
    def double(self) -> SourceLightGCN:
        super().double()
        self.Graph = self.Graph.double()
        return self


def load_train_model() -> SourceLightGCN:
    dataset = dataloader.LastFM(path=str(SOURCE_ROOT / "data" / "lastfm"))
    return SourceLightGCN(world.config, dataset).to(world.device)


def build_optimizer(network: SourceLightGCN) -> torch.optim.Adam:
    return torch.optim.Adam(network.parameters(), lr=0.001)


def train_step(
    network: SourceLightGCN,
    optimizer: torch.optim.Adam,
    users: torch.Tensor,
    positive_items: torch.Tensor,
    negative_items: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad()
    bpr, regularization = network.bpr_loss(users, positive_items, negative_items)
    loss = bpr + 1e-4 * regularization
    loss.backward()
    optimizer.step()
    return loss
