import argparse
from pathlib import Path

from config import get_cfg
from registry import build_model


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=Path, required=True)
parser.add_argument("--opts", default=["MODEL.WIDTH", "4"])
args = parser.parse_args()

cfg = get_cfg()
cfg.merge_from_file(args.cfg)
cfg.merge_from_list(args.opts)

model = build_model(cfg.MODEL)
