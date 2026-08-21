import argparse

from torchvision.models import get_model


parser = argparse.ArgumentParser()
parser.add_argument("--arch", required=True)
parser.add_argument("--classes", type=int, required=True)
args = parser.parse_args()

model = get_model(args.arch, weights=None, num_classes=args.classes)
