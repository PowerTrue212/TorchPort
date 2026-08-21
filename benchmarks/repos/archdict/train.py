import argparse

import models


parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="smallnet")
parser.add_argument("--width", type=int, default=4)
parser.add_argument("--hidden", type=int, default=6)
args = parser.parse_args()

model = models.__dict__[args.arch](width=args.width, hidden=args.hidden)
