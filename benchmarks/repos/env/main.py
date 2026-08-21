import os

import torch
from torch import nn
from torch.nn.functional import relu


WIDTH = int(os.environ["TP_WIDTH"])
HIDDEN = int(os.getenv("TP_HIDDEN", "6"))


class EnvNet(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, value):
        return self.fc2(relu(self.fc1(value)))


def main():
    model = EnvNet(WIDTH, hidden=HIDDEN)
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    print(model(value).shape)


if __name__ == "__main__":
    main()
