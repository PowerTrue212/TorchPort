"""Fetch only selected official Sintel training files through HTTP byte ranges."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/RAFT-Benchmark-v1/deps")
from remotezip import RemoteZip  # noqa: E402

ROOT = Path("/root/autodl-tmp/RAFT-Benchmark-v1")
BASE = "https://files.is.tue.mpg.de/sintel/"
ARCHIVES = {
    "images": "MPI-Sintel-training_images.zip",
    "extras": "MPI-Sintel-training_extras.zip",
}


def main() -> None:
    inventory = {}
    for kind, name in ARCHIVES.items():
        with RemoteZip(BASE + name) as archive:
            names = archive.namelist()
            selected = [item for item in names if
                        ("/clean/alley_1/frame_000" in item if kind == "images"
                         else "/flow/alley_1/frame_000" in item)]
            inventory[kind] = selected[:12]
            if len(selected) < 5:
                raise RuntimeError(f"expected alley_1 files in {name}: {selected}")
            for item in selected[:5]:
                dest = ROOT / "data/sintel-subset" / item
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.read(item))
    manifest = {}
    for path in sorted((ROOT / "data/sintel-subset").rglob("*")):
        if path.is_file():
            manifest[str(path.relative_to(ROOT))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    (ROOT / "evidence/sintel-subset.json").write_text(
        json.dumps({"source": BASE, "inventory": inventory, "files": manifest}, indent=2) + "\n"
    )
    print(json.dumps({"inventory": inventory, "files": manifest}, indent=2))


if __name__ == "__main__":
    main()
