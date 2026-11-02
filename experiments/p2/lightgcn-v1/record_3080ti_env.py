"""Print an isolated 3080 Ti environment record without credentials."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    packages = {name: version(name) for name in (
        "torch", "jittor", "numpy", "PyYAML", "pandas", "scipy", "nvidia-cudnn-cu12", "triton",
    )}
    print(json.dumps({"python": sys.version, "executable": sys.executable,
                      "platform": platform.platform(), "gpu": gpu,
                      "packages": packages}, indent=2))


if __name__ == "__main__":
    main()
