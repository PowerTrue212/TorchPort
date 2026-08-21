import argparse

import timm


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--classes", type=int, required=True)
parser.add_argument("--channels", type=int, required=True)
args = parser.parse_args()

model = timm.create_model(
    args.model,
    in_chans=args.channels,
    num_classes=args.classes,
)
