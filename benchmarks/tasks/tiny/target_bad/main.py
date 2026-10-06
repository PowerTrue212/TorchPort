from jittor import nn


class TinyNet(nn.Module):
    """负例：relu 被故意写成 leaky_relu，验收必须失败。"""

    def __init__(self, width=8, classes=4):
        super().__init__()
        self.fc1 = nn.Linear(width, 16)
        self.fc2 = nn.Linear(16, classes)

    def execute(self, value):
        return nn.softmax(self.fc2(nn.leaky_relu(self.fc1(value), 0.3)), dim=-1)
