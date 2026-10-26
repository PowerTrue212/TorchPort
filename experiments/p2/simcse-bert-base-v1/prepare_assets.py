"""Download the pinned public SimCSE pilot inputs outside candidate workspaces."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import shutil
import time
import urllib.request
from pathlib import Path

ASSETS = (
    ("models", "princeton-nlp/unsup-simcse-bert-base-uncased",
     "218baf6bc48c463a53f5b4e1c568299537fab124",
     ("config.json", "pytorch_model.bin", "tokenizer_config.json",
      "special_tokens_map.json", "vocab.txt")),
    ("models", "google-bert/bert-base-uncased",
     "86b5e0934494bd15c9632b12f734a8a67f723594",
     ("config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.txt")),
    ("datasets", "princeton-nlp/datasets-for-simcse",
     "e145e8bb659b2aa2669f32ef79cb4cdef6c58fef",
     ("senteval.tar", "wiki1m_for_simcse.txt")),
)


def read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download(url: str, target: Path, expected: dict) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".partial")
    if target.exists():
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if target.stat().st_size == expected["size"] and (
            expected["sha256"] is None or digest.hexdigest() == expected["sha256"]
        ):
            return {"path": str(target), "size": target.stat().st_size,
                    "sha256": digest.hexdigest(), "url": url}
        raise ValueError(f"Existing file does not match metadata: {target}")
    for attempt in range(12):
        offset = temp.stat().st_size if temp.exists() else 0
        if offset > expected["size"]:
            raise ValueError(f"Partial file exceeds expected size: {temp}")
        if offset == expected["size"]:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    raise ValueError(f"Server refused byte-range resume: {url}")
                if offset and not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise ValueError(f"Wrong byte range in response: {url}")
                with temp.open("ab") as out:
                    while block := response.read(1024 * 1024):
                        out.write(block)
        except (OSError, TimeoutError, http.client.IncompleteRead) as exc:
            print(f"Retry {attempt + 1}: {type(exc).__name__}", flush=True)
            time.sleep(min(attempt + 1, 10))
            continue
    digest = hashlib.sha256()
    with temp.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if temp.stat().st_size != expected["size"]:
        raise ValueError(f"Wrong size: {temp}")
    if expected["sha256"] is not None and digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"Wrong SHA-256: {temp}")
    shutil.move(temp, target)
    return {"path": str(target), "size": target.stat().st_size,
            "sha256": digest.hexdigest(), "url": url}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("D:/科研/SimCSE-Benchmark-v1/assets"))
    args = parser.parse_args()
    root = args.root.resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "files": []}
    for kind, repo, revision, filenames in ASSETS:
        api = f"https://huggingface.co/api/{kind}/{repo}/revision/{revision}?blobs=true"
        metadata = read_json(api)
        if metadata["sha"] != revision:
            raise ValueError(f"Revision changed: {repo}")
        metadata_path = root / "metadata" / f"{repo.replace('/', '--')}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        siblings = {entry["rfilename"]: entry for entry in metadata["siblings"]}
        for filename in filenames:
            sibling = siblings[filename]
            expected = {"size": sibling["size"],
                        "sha256": sibling.get("lfs", {}).get("sha256")}
            prefix = "datasets/" if kind == "datasets" else ""
            url = f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{filename}"
            target = root / kind / repo.replace("/", "--") / filename
            print(f"Preparing {repo}/{filename} ({expected['size']} bytes)", flush=True)
            manifest["files"].append(download(url, target, expected))
            (root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
