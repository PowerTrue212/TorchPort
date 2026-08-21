from torch import nn


class Tiny(nn.Module):
    def __init__(self, in_chans=3, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, num_classes, 1)

    def forward(self, x):
        return self.conv(x)


def create_model(model_name, in_chans=3, num_classes=2):
    if model_name != "tiny":
        raise ValueError(model_name)
    return Tiny(in_chans=in_chans, num_classes=num_classes)
