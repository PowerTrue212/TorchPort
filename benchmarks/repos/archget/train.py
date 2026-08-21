import argparse

import models


parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="smallnet")
parser.add_argument("--width", type=int, default=4)
parser.add_argument("--hidden", type=int, default=6)
args = parser.parse_args()

builder = getattr(models, args.arch)
model = builder(width=args.width, hidden=args.hidden)
