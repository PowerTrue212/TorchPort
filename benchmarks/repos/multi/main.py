import torch
from torch import nn


class PairNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.left = nn.Linear(width, width)
        self.right = nn.Linear(width, width)
        self.merge = nn.Linear(width * 2, 2)

    def forward(self, left, right):
        left = self.left(left)
        right = self.right(right)
        return self.merge(torch.cat((left, right), dim=1))


def build(width=4):
    return PairNet(width)


def main():
    model = build()
    left = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    right = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    print(model(left, right).shape)


if __name__ == "__main__":
    main()
