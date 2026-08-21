import torch
from torch import nn
import torch.nn.functional as F


class Config:
    width = 4
    hidden = 6


class DynNet(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, value):
        value = F.relu(self.fc1(value))
        return self.fc2(value)


def main():
    config = Config()
    model = DynNet(config.width, config.hidden)
    value = torch.randn(1, config.width)
    print(model(value).shape)


if __name__ == "__main__":
    main()
