"""Remove remote files only after a verified local evidence archive exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path, PurePosixPath

from archive_remote_evidence import ALLOWED_REMOTE_ROOTS, remote_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    manifest_path = archive.with_suffix(archive.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    remote_root = str(PurePosixPath(manifest["remote_root"]).parent)
    digest = PurePosixPath(manifest["remote_root"]).name
    if (remote_root not in ALLOWED_REMOTE_ROOTS
            or manifest["remote_root"] != f"{remote_root}/{digest}"
            or not re.fullmatch(r"[0-9a-f]{12}", digest)):
        raise ValueError("Manifest remote root is outside the SimCSE evidence directory")
    if manifest["archive"] != str(archive) or archive.stat().st_size != manifest["archive_bytes"]:
        raise ValueError("Local archive metadata differs from its manifest")
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["archive_sha256"]:
            raise ValueError("Local archive SHA-256 differs from its manifest")
    expected = manifest["files"]
    for relative in expected:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path in manifest: {relative}")

    from ssh_exec import connect  # noqa: PLC0415

    client = connect()
    try:
        if remote_hashes(client, digest, remote_root) != expected:
            raise RuntimeError("Remote files changed since the local archive was made")
        paths = [f"{remote_root}/{digest}/{relative}" for relative in expected]
        command = "rm -- " + " ".join(shlex.quote(path) for path in paths)
        _, stdout, stderr = client.exec_command(command)
        error = stderr.read().decode("utf-8", "replace")
        stdout.read()
        if stdout.channel.recv_exit_status():
            raise RuntimeError(f"Remote file removal failed: {error}")
    finally:
        client.close()
    print(json.dumps({"archive": str(archive), "removed_remote_files": len(expected)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
