from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import relu


class NestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.merge = nn.Linear(8, 2)

    def forward(self, items: tuple[torch.Tensor, torch.Tensor]):
        left = relu(self.left(items[0]))
        right = relu(self.right(items[1]))
        logits = self.merge(torch.cat((left, right), dim=1))
        return {"logits": logits, "parts": (left, right)}


def main():
    model = NestNet()
    left = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    right = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    print(model((left, right))["logits"].shape)


if __name__ == "__main__":
    main()
