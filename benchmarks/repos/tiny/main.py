import torch
from torch import nn


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, value):
        return self.fc(value)


def main():
    torch.manual_seed(7)
    model = TinyNet()
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    print(model(value).numpy().shape)


if __name__ == "__main__":
    main()
