from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use("Agg")

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "scripts"))

from visuals import draw_summary

folder = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / "out" / "visuals.csv").open()))
fig = draw_summary(rows)
fig.savefig(folder / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
