from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "remote" / "stage_agent_dir.py"


class _Channel:
    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code

    def recv_exit_status(self) -> int:
        return self.exit_code

    def shutdown_write(self) -> None:
        pass


class _Reader(io.BytesIO):
    def __init__(self, content: bytes = b"", exit_code: int = 0) -> None:
        super().__init__(content)
        self.channel = _Channel(exit_code)


class _Writer(io.BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.channel = _Channel()


class _Client:
    def __init__(self, missing: str | None = None) -> None:
        self.missing = missing
        self.commands: list[str] = []
        self.writers: dict[str, _Writer] = {}
        self.closed = False

    def exec_command(
        self, command: str, timeout: float = 0
    ) -> tuple[_Writer, _Reader, _Reader]:
        del timeout
        self.commands.append(command)
        writer = _Writer()
        self.writers[command] = writer
        if self.missing and command == f"test -f {self.missing}":
            return writer, _Reader(exit_code=1), _Reader()
        if command.startswith("sha256sum "):
            return writer, _Reader(b"a" * 64 + b"  file\n"), _Reader()
        if "numpy as np" in command:
            return writer, _Reader(b'{"shape": [1, 3, 8, 8], "dtype": "float32"}\n'), _Reader()
        return writer, _Reader(), _Reader()

    def close(self) -> None:
        self.closed = True


def _load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: _Client
) -> Any:
    monkeypatch.setenv("TP_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("TP_REMOTE_ROOT", "/tp")
    monkeypatch.setenv("TP_REMOTE_SOURCE_PYTHON", "/source/python")

    ssh_exec = types.ModuleType("ssh_exec")
    ssh_exec.connect = lambda: client
    monkeypatch.setitem(sys.modules, "ssh_exec", ssh_exec)

    module_name = f"test_stage_agent_dir_{id(tmp_path)}_{id(client)}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.bundle, "remote_base", lambda task, settings: PurePosixPath("/tp/task-hash"))
    monkeypatch.setattr(
        module.bundle,
        "local_weights",
        lambda task, raw, settings: (tmp_path / raw["weights"], "weight-sha"),
    )
    return module


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "inputs": [{"path": "input.npy"}],
                "weights": "model.pth",
            }
        ),
        encoding="utf-8",
    )
    return task


def test_stage_uses_hard_links_and_writes_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _Client()
    stage = _load(monkeypatch, tmp_path, client)

    assert stage.main(["run-0", str(_task(tmp_path))]) == 0

    staged = "/root/autodl-tmp/agent/run-0/staged"
    assert (
        f"ln -f /tp/task-hash/inputs/0-input.npy {staged}/input.npy" in client.commands
    )
    assert (
        f"ln -f /tp/weights-cache/weight-sha/model.pth {staged}/model.pth"
        in client.commands
    )
    readme_command = f"cat > {staged}/README.txt"
    manifest = client.writers[readme_command].getvalue().decode("utf-8")
    assert "input.npy\tshape=1x3x8x8\tdtype=float32\tsha256=" in manifest
    assert "model.pth\tshape=-\tsha256=" in manifest
    assert "/tp/" not in manifest
    assert client.closed


def test_stage_reports_missing_remote_material_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = "/tp/task-hash/inputs/0-input.npy"
    client = _Client(missing)
    stage = _load(monkeypatch, tmp_path, client)

    assert stage.main(["run-0", str(_task(tmp_path))]) == 1
    output = capsys.readouterr().out
    assert "torchport_inspect" in output
    assert "verify" in output
    assert not any(command.startswith("mkdir -p") for command in client.commands)
    assert client.closed
