from yacs.config import CfgNode as CN


def get_cfg():
    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.TYPE = "tiny"
    cfg.MODEL.WIDTH = 2
    cfg.MODEL.HIDDEN = 3
    return cfg
