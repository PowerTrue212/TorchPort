"""Package the immutable five-step source task without deleting remote files."""
import sys
from dataclasses import replace
from pathlib import Path, PurePosixPath

from torchport.contracts import load_task

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path("D:/科研/FNO-Benchmark-v1")
TASK = BENCH / "controller/train-five-4090-r1/task.json"
SERIES = sys.argv[1]
if not SERIES.isalnum():
    raise ValueError("series must be alphanumeric")
BASE = PurePosixPath(f"/root/autodl-tmp/FNO-Benchmark-v1/oracle-bundle-{SERIES}")

sys.path.insert(0, str(ROOT / "scripts/remote"))
from torchport_remote import bundle  # noqa: E402
from torchport_remote.config import load_settings  # noqa: E402

load_task(TASK)
settings = replace(
    load_settings(ROOT),
    root=PurePosixPath(f"/root/autodl-tmp/FNO-Benchmark-v1/oracle-packages-{SERIES}"),
    source_python="/root/autodl-tmp/envs/source/bin/python",
    target_python="/root/autodl-tmp/envs/target/bin/python",
    python_config="/root/autodl-tmp/envs/target/bin/python3.9-config",
    system_libstdcxx="",
    target_path="/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    extra_roots=(BENCH,),
)
archive = BENCH / f"controller/train-five-4090-r1/oracle-bundle-{SERIES}.tar.gz"
with archive.open("xb") as stream:
    stream.write(bundle.build_bundle(TASK, BASE, settings))
print(archive)
