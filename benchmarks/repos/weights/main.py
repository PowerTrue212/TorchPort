import torch
from torch import nn
from torch.nn.functional import relu


class WeightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, value):
        return self.fc2(relu(self.fc1(value)))


def main():
    model = WeightNet()
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    print(model(value).shape)


if __name__ == "__main__":
    main()
