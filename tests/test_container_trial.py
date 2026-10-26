from __future__ import annotations

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


def test_run_args_mounts_only_workspace_and_listed_paths() -> None:
    args = trial.run_args(
        "img:1", "tp-x", "/mnt/d/ws", [("/mnt/d/c/patch.yml", "/home/node/p.yml", True)],
        ["DEEPSEEK_API_KEY"], ["dsh", "--json"], interactive=True,
    )
    volumes = [args[i + 1] for i, item in enumerate(args) if item == "-v"]
    assert volumes == ["/mnt/d/ws:/work", "/mnt/d/c/patch.yml:/home/node/p.yml:ro"]
    # 只传变量名，密钥值不出现在命令行
    assert args[args.index("-e") + 1] == "DEEPSEEK_API_KEY"
    assert "--rm" in args and "-i" in args
    assert args[-3:] == ["img:1", "dsh", "--json"]
