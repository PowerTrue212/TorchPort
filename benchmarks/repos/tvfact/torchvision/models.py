from torch import nn


class Tiny(nn.Module):
    def __init__(self, in_chans=3, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, num_classes, 1)

    def forward(self, x):
        return self.conv(x)


def get_model(name, weights=None, num_classes=2):
    if name != "tiny":
        raise ValueError(name)
    del weights
    return Tiny(in_chans=3, num_classes=num_classes)
