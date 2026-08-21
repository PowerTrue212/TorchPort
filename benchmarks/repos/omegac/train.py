import argparse

from model import Net
from omegaconf import OmegaConf


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

cfg = OmegaConf.load(args.config)
options = OmegaConf.to_container(cfg.model, resolve=True)
model = Net(**options)
