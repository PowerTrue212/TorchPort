from torch import nn


class Net(nn.Module):
    def __init__(self, width=4, hidden=6):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


MODELS = {"tiny": Net}


def build_model(cfg):
    spec = dict(cfg)
    kind = spec.pop("TYPE")
    model_cls = MODELS[kind]
    return model_cls(width=spec["WIDTH"], hidden=spec["HIDDEN"])
