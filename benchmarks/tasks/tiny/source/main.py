from torch import nn
from torch.nn import functional as F


class TinyNet(nn.Module):
    def __init__(self, width=8, classes=4):
        super().__init__()
        self.fc1 = nn.Linear(width, 16)
        self.fc2 = nn.Linear(16, classes)

    def forward(self, value):
        return F.softmax(self.fc2(F.relu(self.fc1(value))), dim=-1)
