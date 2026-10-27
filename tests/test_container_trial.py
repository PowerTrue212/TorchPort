from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from experiments.harness.agents import run_container_trial as trial


def test_windows_path_maps_to_wsl_mount() -> None:
    assert trial.to_docker_path(Path(r"D:\科研\CLIP\workspaces\none"), True) == \
        "/mnt/d/科研/CLIP/workspaces/none"


def test_posix_path_is_unchanged() -> None:
    assert trial.to_docker_path(Path("/srv/work"), False) == "/srv/work"


def test_unc_path_is_rejected() -> None:
    with pytest.raises(ValueError):
        trial.to_docker_path(Path(r"\\server\share\work"), True)


def test_docker_prefix_uses_wsl_on_windows() -> None:
    assert trial.docker_prefix(True, "Ubuntu") == ["wsl.exe", "-d", "Ubuntu", "--", "docker"]
    assert trial.docker_prefix(False, "Ubuntu") == ["docker"]


def test_remove_container_uses_docker_prefix_and_ignores_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    options = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        options.append(_kwargs)
        return subprocess.CompletedProcess(command, 1, "", "No such container: tp-x")

    monkeypatch.setattr(trial.subprocess, "run", fake_run)
    trial.remove_container(trial.docker_prefix(True, "Ubuntu"), "tp-x")
    assert commands == [["wsl.exe", "-d", "Ubuntu", "--", "docker", "rm", "-f", "tp-x"]]
    assert options[0]["encoding"] == "utf-8"
    assert options[0]["errors"] == "replace"


def test_run_args_mounts_only_workspace_and_listed_paths() -> None:
    args = trial.run_args(
        "img:1", "tp-x", "/mnt/d/ws", [("/mnt/d/c/patch.yml", "/home/node/p.yml", True)],
        ["DEEPSEEK_API_KEY"], ["dsh", "--json"], interactive=True,
    )
    volumes = [args[i + 1] for i, item in enumerate(args) if item == "-v"]
    assert volumes == [
        "/mnt/d/ws:/work", "/work/.scratch", "/mnt/d/c/patch.yml:/home/node/p.yml:ro",
    ]
    # 只传变量名，密钥值不出现在命令行
    assert args[args.index("-e") + 1] == "DEEPSEEK_API_KEY"
    assert "--rm" in args and "-i" in args
    assert args[-3:] == ["img:1", "dsh", "--json"]


def test_memory_limit_sets_equal_swap_and_is_optional() -> None:
    args = trial.run_args("img:1", "tp-x", "/mnt/d/ws", [], [], ["true"], memory="5g")
    assert args[args.index("--memory") + 1] == "5g"
    assert args[args.index("--memory-swap") + 1] == "5g"
    assert "--memory" not in trial.run_args("img:1", "tp-x", "/mnt/d/ws", [], [], ["true"])


def test_stop_reason_precedence() -> None:
    # 越界优先于基础设施错误；清理失败不归因于 Agent
    assert trial.choose_stop_reason(True, 0, True) == "scope_violation"
    assert trial.choose_stop_reason(False, 0, True) == "infrastructure_error"
    assert trial.choose_stop_reason(False, 0, False) == "agent_completed"
    assert trial.choose_stop_reason(False, 1, False) == "agent_error"
    assert trial.choose_stop_reason(False, None, False) == "agent_error"


def test_proxy_env_bypasses_loopback_and_is_empty_without_proxy() -> None:
    assert trial.proxy_env(None) == []
    env = trial.proxy_env("http://127.0.0.1:7890")
    assert "HTTPS_PROXY=http://127.0.0.1:7890" in env
    assert "NO_PROXY=localhost,127.0.0.1,::1" in env
    assert "NODE_USE_ENV_PROXY=1" in env
