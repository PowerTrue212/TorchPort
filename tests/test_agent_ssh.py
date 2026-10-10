from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "remote" / "agent_ssh.py"
ALLOWED = "/root/autodl-tmp/agent/plantcamo-full-r0"


class _Channel:
    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self.write_closed = False

    def recv_exit_status(self) -> int:
        return self.exit_code

    def shutdown_write(self) -> None:
        self.write_closed = True


class _Reader(io.BytesIO):
    def __init__(self, content: bytes = b"", exit_code: int = 0) -> None:
        super().__init__(content)
        self.channel = _Channel(exit_code)


class _Writer(io.BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.channel = _Channel()


class _Client:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.writers: dict[str, _Writer] = {}
        self.closed = False

    def exec_command(self, command: str) -> tuple[_Writer, _Reader, _Reader]:
        self.commands.append(command)
        writer = _Writer()
        self.writers[command] = writer
        stdout = _Reader(b"ok")
        return writer, stdout, _Reader()

    def close(self) -> None:
        self.closed = True


def _load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: _Client | None = None
) -> tuple[Any, list[int]]:
    calls: list[int] = []

    def connect() -> _Client:
        calls.append(1)
        if client is None:
            raise AssertionError("must not connect")
        return client

    ssh_exec = types.ModuleType("ssh_exec")
    ssh_exec.connect = connect
    monkeypatch.setitem(sys.modules, "ssh_exec", ssh_exec)
    monkeypatch.setenv("TP_AGENT_AUDIT", str(tmp_path / "audit.jsonl"))
    module_name = f"test_agent_ssh_{id(tmp_path)}_{id(client)}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, calls


@pytest.mark.parametrize(
    "command",
    [
        "ls tp-work",
        "cat EVIDENCE/result.json",
        "ls weights-cache",
        "find oracle -type f",
        "cat noise-envelope.json",
        "cat ~/.tp_ssh.env",
        "ls /root/autodl-tmp/heldout",
        "ls /root/autodl-tmp/tp-secret",
    ],
)
def test_forbidden_substring_is_rejected_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    monkeypatch.setenv("TP_AGENT_REMOTE_DIR", ALLOWED)
    wrapper, calls = _load(monkeypatch, tmp_path)

    assert wrapper.main(["run", command]) == 2
    assert calls == []
    assert capsys.readouterr().out.startswith("拒绝执行：")


def test_other_agent_directory_is_rejected_without_connecting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TP_AGENT_REMOTE_DIR", ALLOWED)
    wrapper, calls = _load(monkeypatch, tmp_path)

    assert wrapper.main(["run", "ls /root/autodl-tmp/agent/another-run"]) == 2
    assert calls == []


@pytest.mark.parametrize(
    "command",
    [
        # VSCode-v2 full-r0 的两条真实误拦：转义引号紧跟在自己的目录名后面
        f"python -c 'import sys; sys.path.insert(0,\" {ALLOWED}\\\");'",
        f"ls {ALLOWED}/ && grep -n 'x' {ALLOWED}`echo`",
    ],
)
def test_own_directory_followed_by_escape_is_not_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str
) -> None:
    monkeypatch.setenv("TP_AGENT_REMOTE_DIR", ALLOWED)
    client = _Client()
    wrapper, calls = _load(monkeypatch, tmp_path, client)

    assert wrapper.main(["run", command]) == 0
    assert calls == [1]


def test_allowed_command_is_prefixed_with_remote_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TP_AGENT_REMOTE_DIR", ALLOWED)
    client = _Client()
    wrapper, calls = _load(monkeypatch, tmp_path, client)

    command = f"python smoke.py {ALLOWED}/staged/input.npy"
    assert wrapper.main(["run", command]) == 0
    assert calls == [1]
    assert client.commands == [f"cd {ALLOWED} && {command}"]
    assert client.closed


@pytest.mark.parametrize("remote", ["../x", "/root/autodl-tmp/outside/x"])
def test_put_path_traversal_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote: str
) -> None:
    monkeypatch.setenv("TP_AGENT_REMOTE_DIR", ALLOWED)
    local = tmp_path / "submission.py"
    local.write_text("pass\n", encoding="utf-8")
    wrapper, calls = _load(monkeypatch, tmp_path)

    assert wrapper.main(["put", str(local), remote]) == 2
    assert calls == []


def test_put_streams_file_to_normalized_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TP_AGENT_REMOTE_DIR", ALLOWED)
    local = tmp_path / "submission.py"
    local.write_bytes(b"print('ok')\n")
    client = _Client()
    wrapper, calls = _load(monkeypatch, tmp_path, client)

    assert wrapper.main(["put", str(local), "nested/../submission.py"]) == 0

    command = f"mkdir -p {ALLOWED} && cat > {ALLOWED}/submission.py"
    assert calls == [1]
    assert client.commands == [command]
    assert client.writers[command].getvalue() == b"print('ok')\n"
    assert client.writers[command].channel.write_closed


def test_audit_records_accepted_and_rejected_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TP_AGENT_REMOTE_DIR", ALLOWED)
    client = _Client()
    wrapper, _ = _load(monkeypatch, tmp_path, client)

    assert wrapper.main(["run", "pwd"]) == 0
    assert wrapper.main(["run", "find tp-work"]) == 2

    records = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text("utf-8").splitlines()]
    assert [(item["accepted"], item["exit_code"]) for item in records] == [
        (True, 0),
        (False, 2),
    ]
    assert records[0]["command"] == "pwd"
    assert records[1]["command"] == "find tp-work"
    assert all(item["time"] for item in records)


def test_refuses_to_start_without_allowed_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TP_AGENT_REMOTE_DIR", raising=False)
    wrapper, calls = _load(monkeypatch, tmp_path)

    assert wrapper.main(["run", "pwd"]) == 2
    assert calls == []
    assert "TP_AGENT_REMOTE_DIR" in capsys.readouterr().out
    record = json.loads((tmp_path / "audit.jsonl").read_text("utf-8"))
    assert record["accepted"] is False
    assert record["exit_code"] == 2
    assert record["command"] == "pwd"
