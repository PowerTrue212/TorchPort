from jittor import nn


class TinyNet(nn.Module):
    def __init__(self, width=8, classes=4):
        super().__init__()
        self.fc1 = nn.Linear(width, 16)
        self.fc2 = nn.Linear(16, classes)

    def execute(self, value):
        return nn.softmax(self.fc2(nn.relu(self.fc1(value))), dim=-1)
