import argparse

import torch
from torch import nn
from torch.nn.functional import relu


class ArgNet(nn.Module):
    def __init__(self, width, hidden, classes, enabled):
        super().__init__()
        self.enabled = enabled
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, value):
        return self.fc2(relu(self.fc1(value)))


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default="4")
    parser.add_argument("--hidden-size", dest="hidden", type=int, default=6)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--enabled", action="store_true")
    return parser.parse_args([])


def main():
    args = options()
    model = ArgNet(
        args.width,
        hidden=args.hidden,
        classes=args.classes,
        enabled=args.enabled,
    )
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    print(model(value).shape)


if __name__ == "__main__":
    main()
