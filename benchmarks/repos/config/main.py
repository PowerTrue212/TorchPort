import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn.functional import relu


class ConfigNet(nn.Module):
    def __init__(self, width, hidden, classes):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, value):
        return self.fc2(relu(self.fc1(value)))


def main():
    text = Path(__file__).with_name("model.json").read_text(encoding="utf-8")
    config = json.loads(text)
    text = Path(__file__).with_name("model.yaml").read_text(encoding="utf-8")
    extra = yaml.safe_load(text)
    options = {**config["model"], **extra["model"]}
    model = ConfigNet(**options)
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    print(model(value).shape)


if __name__ == "__main__":
    main()
