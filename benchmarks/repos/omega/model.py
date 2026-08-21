from torch import nn


class Net(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, value):
        return self.fc2(self.fc1(value))
