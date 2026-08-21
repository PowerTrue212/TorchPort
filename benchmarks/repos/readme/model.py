from torch import nn


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(1, upscale_factor, 3)

    def forward(self, value):
        return self.conv(value)
