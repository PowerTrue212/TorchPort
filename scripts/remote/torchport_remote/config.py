"""远端 TorchPort 适配器的环境配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REMOTE_ENV_NAMES = {
    "TP_REMOTE_ROOT", "TP_REMOTE_SOURCE_PYTHON", "TP_REMOTE_TARGET_PYTHON",
    "TP_REMOTE_PYTHON_CONFIG", "TP_REMOTE_LIBSTDCXX", "TP_REMOTE_TARGET_PATH",
    "TP_REMOTE_NVCC_PATH", "TP_REMOTE_USE_CUTT", "TP_REMOTE_USE_MKL",
}


def load_env_file(path: str | Path | None = None) -> None:
    """只加载远端路径设置，不读取或暴露其他变量。"""
    path = Path(path or os.environ.get("TP_SSH_ENV_FILE") or Path.home() / ".tp_ssh.env")
    if not path.is_file():
        return
    for raw in path.read_text("utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name in REMOTE_ENV_NAMES:
            os.environ.setdefault(name, value)


@dataclass(frozen=True)
class RemoteSettings:
    root: PurePosixPath
    source_python: str
    target_python: str
    python_config: str
    system_libstdcxx: str
    target_path: str
    target_env_defaults: dict[str, str | None]
    project_root: Path
    extra_roots: tuple[Path, ...]
    feedback: str

    @property
    def weights_cache(self) -> PurePosixPath:
        return self.root / "weights-cache"


def load_settings(default_project_root: Path) -> RemoteSettings:
    load_env_file()
    return RemoteSettings(
        root=PurePosixPath(os.environ.get("TP_REMOTE_ROOT", "/root/autodl-tmp/tp-dsh-web")),
        source_python=os.environ.get("TP_REMOTE_SOURCE_PYTHON", "/root/miniconda3/bin/python"),
        target_python=os.environ.get("TP_REMOTE_TARGET_PYTHON", "/root/autodl-tmp/jt-env/bin/python"),
        python_config=os.environ.get("TP_REMOTE_PYTHON_CONFIG", "/root/miniconda3/bin/python3.12-config"),
        system_libstdcxx=os.environ.get("TP_REMOTE_LIBSTDCXX", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"),
        target_path=os.environ.get(
            "TP_REMOTE_TARGET_PATH",
            "/usr/local/cuda/bin:/root/autodl-tmp/jt-env/bin:/usr/local/sbin:"
            "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        ),
        target_env_defaults={
            "use_cutt": os.environ.get("TP_REMOTE_USE_CUTT"),
            "use_mkl": os.environ.get("TP_REMOTE_USE_MKL"),
            "nvcc_path": os.environ.get("TP_REMOTE_NVCC_PATH"),
        },
        project_root=Path(os.environ.get("TP_PROJECT_ROOT") or default_project_root).resolve(),
        extra_roots=tuple(
            Path(item).resolve()
            for item in os.environ.get("TP_EXTRA_ROOTS", "").split(os.pathsep)
            if item
        ),
        feedback=os.environ.get("TP_FEEDBACK", "localized"),
    )
