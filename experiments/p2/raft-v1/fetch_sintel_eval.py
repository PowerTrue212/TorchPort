"""Fetch two held-out Sintel alley_1 frame pairs from the official image archive."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/RAFT-Benchmark-v1")
sys.path.insert(0, str(ROOT / "deps"))
from remotezip import RemoteZip  # noqa: E402

url = "https://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip"
names = [f"training/clean/alley_1/frame_{n:04d}.png" for n in (6, 7, 8)]
with RemoteZip(url) as archive:
    for name in names:
        path = ROOT / "data/sintel-subset" / name
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
flow_name = "training/flow/alley_1/frame_0006.flo"
with RemoteZip("https://files.is.tue.mpg.de/sintel/MPI-Sintel-training_extras.zip") as archive:
    path = ROOT / "data/sintel-subset" / flow_name
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read(flow_name))
names.append(flow_name)
manifest = {name: hashlib.sha256((ROOT / "data/sintel-subset" / name).read_bytes()).hexdigest()
            for name in names}
(ROOT / "evidence/sintel-eval-r1.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
