import argparse

from omegaconf import OmegaConf

from model import Net


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

cfg = OmegaConf.load(args.config)
model = Net(**cfg.model)
