from torch import nn
from torch.nn.functional import relu


class PackNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, 2)

    def forward(self, value):
        return self.fc2(relu(self.fc1(value)))
