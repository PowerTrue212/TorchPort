"""Controller-only Jittor LightGCN training reference for the LastFM pilot."""

from __future__ import annotations

from pathlib import Path

import jittor as jt
from jittor import nn

USERS = 1892
ITEMS = 4489
DIM = 64
LAYERS = 3


class LightGCNReference(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_user = nn.Embedding(USERS, DIM)
        self.embedding_item = nn.Embedding(ITEMS, DIM)
        import numpy as np  # noqa: PLC0415

        with np.load(Path(__file__).with_name("graph-r1.npz"), allow_pickle=False) as graph:
            indices = jt.array(graph["indices"].astype(np.int32))
            values = jt.array(graph["values"])
            shape = jt.NanoVector(graph["shape"].tolist())
        self.graph = jt.sparse.sparse_array(indices, values, shape)

    def computer(self) -> tuple[jt.Var, jt.Var]:
        combined = jt.concat([self.embedding_user.weight, self.embedding_item.weight], dim=0)
        layers = [combined]
        for _ in range(LAYERS):
            combined = jt.sparse.spmm(self.graph, combined)
            layers.append(combined)
        result = jt.stack(layers, dim=1).mean(dim=1)
        return result[:USERS], result[USERS:]

    def bpr_loss(self, users: jt.Var, positive: jt.Var, negative: jt.Var) -> jt.Var:
        user_table, item_table = self.computer()
        user = user_table[users]
        pos = item_table[positive]
        neg = item_table[negative]
        pos_score = (user * pos).sum(dim=1)
        neg_score = (user * neg).sum(dim=1)
        bpr = nn.softplus(neg_score - pos_score).mean()
        u0 = self.embedding_user(users)
        p0 = self.embedding_item(positive)
        n0 = self.embedding_item(negative)
        regularization = ((u0 * u0).sum() + (p0 * p0).sum() + (n0 * n0).sum()) / (2 * users.shape[0])
        return bpr + 1e-4 * regularization

    def execute(self, users: jt.Var, items: jt.Var) -> jt.Var:
        user_table, item_table = self.computer()
        return (user_table[users] * item_table[items]).sum(dim=1)


def load_train_model() -> LightGCNReference:
    return LightGCNReference()


def build_optimizer(model: LightGCNReference) -> jt.optim.Adam:
    parameters = model.parameters()
    for parameter in parameters:
        parameter.start_grad()
    optimizer = jt.optim.Adam(parameters, lr=0.001)
    optimizer.param_groups[0]["lr"] = 0.001
    return optimizer


def train_step(
    model: LightGCNReference,
    optimizer: jt.optim.Adam,
    users: jt.Var,
    positive_items: jt.Var,
    negative_items: jt.Var,
) -> jt.Var:
    loss = model.bpr_loss(users, positive_items, negative_items)
    optimizer.step(loss)
    return loss
