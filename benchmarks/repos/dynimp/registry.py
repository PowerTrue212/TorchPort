import importlib


def build_model(cfg):
    spec = dict(cfg)
    module = importlib.import_module(spec.pop("module"))
    model_cls = getattr(module, spec.pop("name"))
    return model_cls(**spec)
