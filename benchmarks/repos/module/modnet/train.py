import argparse

from .model import Net


parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, required=True)
parser.add_argument("--hidden", type=int, required=True)
opt = parser.parse_args()

model = Net(width=opt.width, hidden=opt.hidden)
