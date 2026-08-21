from model import Net


MODELS = {"tiny": Net}


def build_model(cfg):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    model_cls = MODELS[kind]
    return model_cls(**cfg)
