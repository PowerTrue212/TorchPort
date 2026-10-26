"""Copy one completed remote evidence tree to a verified local tar archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/remote"))
from ssh_exec import connect  # noqa: E402

REMOTE_ROOT = "/root/data-fs/SimCSE-Benchmark-v1/harness-evidence"
ALLOWED_REMOTE_ROOTS = {
    REMOTE_ROOT,
    "/root/data-fs/SimCSE-Benchmark-v1/harness",
}


def remote_hashes(client, digest: str, remote_root: str = REMOTE_ROOT) -> dict[str, str]:
    if remote_root not in ALLOWED_REMOTE_ROOTS or not re.fullmatch(r"[0-9a-f]{12}", digest):
        raise ValueError("Remote evidence path is outside the allowed roots")
    command = (f"cd {remote_root}/{digest} && "
               "find . -type f -print0 | sort -z | xargs -0 -r sha256sum")
    _, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode("utf-8", "replace")
    error = stderr.read().decode("utf-8", "replace")
    if stdout.channel.recv_exit_status():
        raise RuntimeError(f"Remote hash scan failed: {error}")
    hashes = {}
    for line in output.splitlines():
        sha, path = line.split("  ", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or not path.startswith("./"):
            raise ValueError(f"Unexpected remote hash line: {line}")
        hashes[path[2:]] = sha
    if not hashes:
        raise ValueError("Remote evidence tree is empty")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-root", choices=sorted(ALLOWED_REMOTE_ROOTS),
                        default=REMOTE_ROOT)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{12}", args.digest):
        raise ValueError("Digest must be exactly 12 lowercase hex characters")
    output = args.output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")

    client = connect()
    try:
        expected = remote_hashes(client, args.digest, args.remote_root)
        command = f"tar -C {args.remote_root} -cf - {args.digest}"
        _, stdout, stderr = client.exec_command(command)
        archive_hash = hashlib.sha256()
        with partial.open("wb") as stream:
            while block := stdout.read(4 * 1024 * 1024):
                stream.write(block)
                archive_hash.update(block)
        error = stderr.read().decode("utf-8", "replace")
        if stdout.channel.recv_exit_status():
            raise RuntimeError(f"Remote tar failed: {error}")
        if remote_hashes(client, args.digest, args.remote_root) != expected:
            raise RuntimeError("Remote evidence changed during archive transfer")
    finally:
        client.close()

    found = {}
    with tarfile.open(partial, "r:") as archive:
        for member in archive:
            if not (member.isfile() or member.islnk()):
                continue
            prefix = args.digest + "/"
            if not member.name.startswith(prefix):
                raise ValueError(f"Unexpected tar member: {member.name}")
            relative = member.name[len(prefix):]
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Missing tar file body: {member.name}")
            file_hash = hashlib.sha256()
            while block := source.read(4 * 1024 * 1024):
                file_hash.update(block)
            found[relative] = file_hash.hexdigest()
    if found != expected:
        raise RuntimeError("Archive content differs from the remote file hashes")
    partial.replace(output)
    manifest = {"remote_root": f"{args.remote_root}/{args.digest}",
                "archive": str(output), "archive_sha256": archive_hash.hexdigest(),
                "archive_bytes": output.stat().st_size,
                "file_count": len(found), "files": found}
    report = output.with_suffix(output.suffix + ".manifest.json")
    report.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"archive": str(output), "files": len(found),
                      "bytes": output.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
