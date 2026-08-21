import argparse
from pathlib import Path

import yaml

from model import Net


def load_config(path):
    with open(path) as stream:
        return yaml.safe_load(stream)


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=Path, required=True)
args = parser.parse_args()

config = load_config(args.config)
model = Net(**config["model"])
