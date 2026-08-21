import argparse

from .model import Net


parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, required=True)
parser.add_argument("--hidden", type=int, required=True)
args = parser.parse_args()

model = Net(width=args.width, hidden=args.hidden)
