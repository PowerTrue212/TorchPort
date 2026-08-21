from torch import nn


class Registry:
    def __init__(self):
        self.items = {}

    def register_module(self, name=None):
        def wrap(cls):
            self.items[name or cls.__name__] = cls
            return cls

        return wrap

    def __getitem__(self, key):
        return self.items[key]


MODELS = Registry()


@MODELS.register_module()
class Net(nn.Module):
    def __init__(self, width=4, hidden=6):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def build_model(cfg):
    cfg = cfg.copy()
    kind = cfg.pop("type")
    model_cls = MODELS[kind]
    return model_cls(**cfg)
