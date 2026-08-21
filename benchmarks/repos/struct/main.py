import torch
from torch import nn
from torch.nn.functional import relu


class BatchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.merge = nn.Linear(8, 2)

    def forward(self, batch, *, scale):
        left = relu(self.left(batch["left"]))
        right = relu(self.right(batch["right"]))
        value = torch.cat((left, right), dim=1)
        return self.merge(value) * scale


def main():
    model = BatchNet()
    left = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    right = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    scale = torch.tensor([[0.5]])
    print(model({"left": left, "right": right}, scale=scale).shape)


if __name__ == "__main__":
    main()
