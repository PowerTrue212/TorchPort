from torch import nn


class Net(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.conv = nn.Conv2d(1, hidden, 3)
        self.head = nn.Conv2d(hidden, width, 1)

    def forward(self, value):
        return self.head(self.conv(value))
