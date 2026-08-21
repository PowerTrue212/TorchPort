import argparse

from hydra.utils import instantiate
from omegaconf import OmegaConf


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

cfg = OmegaConf.load(args.config)
model = instantiate(cfg.model)
