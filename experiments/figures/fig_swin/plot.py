from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "scripts"))

from visuals import draw

fig = draw("swin")
fig.savefig(Path(__file__).resolve().parent / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
