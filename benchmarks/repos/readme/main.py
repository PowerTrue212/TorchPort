import argparse

from model import Net


parser = argparse.ArgumentParser()
parser.add_argument("--upscale_factor", type=int, required=True)
parser.add_argument("--seed", type=int, default=7)
opt = parser.parse_args()

model = Net(upscale_factor=opt.upscale_factor)
