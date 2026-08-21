import argparse
from pathlib import Path

import yaml

from registry import build_model


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()

with open(args.config) as stream:
    cfg = yaml.safe_load(stream)

model = build_model(cfg["model"])
