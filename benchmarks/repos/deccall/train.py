import argparse

from omegaconf import OmegaConf
from registry import build_model


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

cfg = OmegaConf.load(args.config)
model = build_model(cfg.model)
