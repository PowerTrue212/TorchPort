from __future__ import annotations

import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from common import ROOT, digest, load_config, tree_digest, verify_assets, write_json

CIFAR_FILES = {
    "train": "16019d7e3df5f24257cddd939b257f8d",
    "test": "f0ef6b0ae62326f3e7ffdfab6717acfc",
    "meta": "7973b15100ade9c7d40fb424638fde48",
}


def safe_zip_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe ZIP member: {member.filename}")
        package.extractall(destination)


def safe_tar_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as package:
        for member in package.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe TAR member: {member.name}")
        package.extractall(destination)


def remove_packaging_noise(root: Path) -> None:
    for directory in sorted(root.rglob("__pycache__"), reverse=True):
        if directory.is_dir():
            shutil.rmtree(directory)
    for path in root.rglob("*"):
        if path.is_file() and (path.name == ".DS_Store" or path.suffix == ".pyc"):
            path.unlink()


def replace_once(path: Path, old: str, new: str) -> None:
    content = path.read_text(encoding="utf-8")
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"expected one protocol patch target in {path}, found {count}")
    path.write_text(content.replace(old, new), encoding="utf-8", newline="\n")


def patch_source(source: Path) -> None:
    zoo = source / "models" / "zoo.py"
    replace_once(
        zoo,
        "import math\nfrom operator import mul",
        "import math\nimport os\nfrom pathlib import Path\nfrom operator import mul",
    )
    replace_once(
        zoo,
        "load_dict = vit_base_patch16_224(pretrained=True).state_dict()",
        "checkpoint = Path(os.environ['APT_VIT_PRETRAINED']).expanduser()\n"
        "            if not checkpoint.is_file():\n"
        "                raise FileNotFoundError(checkpoint)\n"
        "            pretrained_model = vit_base_patch16_224(pretrained=False)\n"
        "            pretrained_model.load_pretrained(str(checkpoint))\n"
        "            load_dict = pretrained_model.state_dict()",
    )

    trainer = source / "trainer.py"
    content = trainer.read_text(encoding="utf-8")
    old = "shuffle=False, drop_last=True, num_workers=self.workers"
    count = content.count(old)
    if count != 2:
        raise RuntimeError(f"expected two test-loader patch targets in {trainer}, found {count}")
    trainer.write_text(content.replace(old, "shuffle=False, drop_last=False, num_workers=self.workers"), encoding="utf-8", newline="\n")

    run = source / "run.py"
    replace_once(
        run,
        "seed = 0\nrandom.seed(seed)\nnp.random.seed(seed)\ntorch.manual_seed(seed)\ntorch.cuda.manual_seed(seed)",
        "def seed_everything(seed):\n"
        "    random.seed(seed)\n"
        "    np.random.seed(seed)\n"
        "    torch.manual_seed(seed)\n"
        "    torch.cuda.manual_seed_all(seed)",
    )
    replace_once(
        run,
        "args = get_args(sys.argv[1:])\n    print(args)",
        "args = get_args(sys.argv[1:])\n    seed_everything(args.seed)\n    print(args)",
    )
    replace_once(
        run,
        "torch.backends.cudnn.deterministic=True",
        "torch.backends.cudnn.deterministic=True\n    torch.backends.cudnn.benchmark=False",
    )


def prepare_source(archive: Path, config: dict) -> Path:
    source = ROOT / "source" / "apt"
    if source.exists():
        actual = tree_digest(source)
        expected = config["source"]["protocol_source_tree_sha256"]
        if actual != expected:
            raise RuntimeError(f"source tree changed: expected {expected}, got {actual}")
        return source

    staging = ROOT / "source" / ".apt-extract-part"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    safe_zip_extract(archive, staging)
    extracted = staging / config["source"]["extracted_directory"]
    if not extracted.is_dir():
        raise FileNotFoundError(extracted)
    remove_packaging_noise(extracted)
    patch_source(extracted)
    actual = tree_digest(extracted)
    expected = config["source"]["protocol_source_tree_sha256"]
    if actual != expected:
        raise RuntimeError(f"protocol source tree mismatch: expected {expected}, got {actual}")
    source.parent.mkdir(parents=True, exist_ok=True)
    extracted.replace(source)
    shutil.rmtree(staging)
    return source


def prepare_dataset(archive: Path) -> Path:
    data_root = ROOT / "shared" / "data"
    extracted = data_root / "cifar-100-python"
    valid = all(
        (extracted / name).is_file() and digest(extracted / name, "md5") == expected
        for name, expected in CIFAR_FILES.items()
    )
    if not valid:
        data_root.mkdir(parents=True, exist_ok=True)
        safe_tar_extract(archive, data_root)
    for name, expected in CIFAR_FILES.items():
        path = extracted / name
        if not path.is_file() or digest(path, "md5") != expected:
            raise RuntimeError(f"invalid extracted CIFAR-100 file: {path}")
    return data_root


def prepare_checkpoint(config: dict) -> Path:
    backbone = config["backbone"]
    output = ROOT / "shared" / "checkpoints" / backbone["checkpoint_filename"]
    expected_bytes = int(backbone["checkpoint_bytes"])
    if (
        output.is_file()
        and output.stat().st_size == expected_bytes
        and digest(output, "md5") == backbone["checkpoint_md5"]
    ):
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(
        backbone["checkpoint_url"], headers={"User-Agent": "APT-frozen-protocol/1.0"}
    )
    print(f"downloading frozen ViT checkpoint -> {output}")
    with urllib.request.urlopen(request) as source, partial.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    actual_md5 = digest(partial, "md5")
    if partial.stat().st_size != expected_bytes or actual_md5 != backbone["checkpoint_md5"]:
        size = partial.stat().st_size
        partial.unlink()
        raise RuntimeError(f"checkpoint mismatch: bytes={size}, md5={actual_md5}")
    partial.replace(output)
    return output


def main() -> int:
    config = load_config()
    assets = verify_assets()
    source = prepare_source(assets["source"], config)
    data_root = prepare_dataset(assets["dataset"])
    checkpoint = prepare_checkpoint(config)
    write_json(
        ROOT / "shared" / "provenance" / "bootstrap.json",
        {
            "protocol_version": config["protocol_version"],
            "source_repository": config["source"]["repository"],
            "source_commit": config["source"]["commit"],
            "source_archive_sha256": digest(assets["source"]),
            "source_tree_sha256": tree_digest(source),
            "dataset_archive_sha256": digest(assets["dataset"]),
            "checkpoint_md5": digest(checkpoint, "md5"),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "data_files": {
                name: digest(data_root / "cifar-100-python" / name, "md5")
                for name in CIFAR_FILES
            },
        },
    )
    print(f"APT bootstrap complete; source tree sha256={tree_digest(source)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
