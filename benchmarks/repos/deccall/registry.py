from torch import nn


class Registry:
    def __init__(self):
        self.items = {}

    def register(self, name):
        def wrap(cls):
            self.items[name] = cls
            return cls

        return wrap

    def __getitem__(self, key):
        return self.items[key]


MODELS = Registry()


@MODELS.register("tiny")
class Tiny(nn.Module):
    def __init__(self, width=4, hidden=6):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def build_model(cfg):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    model_cls = MODELS[kind]
    return model_cls(**cfg)
