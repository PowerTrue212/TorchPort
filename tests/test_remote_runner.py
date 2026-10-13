from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import types
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "remote" / "dsh_torchport.py"
REMOTE_ENV_NAMES = (
    "TP_REMOTE_ROOT",
    "TP_REMOTE_SOURCE_PYTHON",
    "TP_REMOTE_TARGET_PYTHON",
    "TP_REMOTE_PYTHON_CONFIG",
    "TP_REMOTE_LIBSTDCXX",
    "TP_REMOTE_TARGET_PATH",
    "TP_REMOTE_NVCC_PATH",
    "TP_REMOTE_USE_CUTT",
    "TP_REMOTE_USE_MKL",
)


def _load_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env: dict[str, str] | None = None,
    env_file: str | None = None,
) -> Any:
    for name in REMOTE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    if env_file is not None:
        home.mkdir()
        (home / ".tp_ssh.env").write_text(env_file, encoding="utf-8")
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    # 打包根目录固定为测试临时目录，与 DSH 调用时的 cwd 无关。
    monkeypatch.setenv("TP_PROJECT_ROOT", str(tmp_path))

    ssh_exec = types.ModuleType("ssh_exec")
    ssh_exec.connect = lambda: None
    monkeypatch.setitem(sys.modules, "ssh_exec", ssh_exec)

    module_name = f"test_dsh_torchport_{id(tmp_path)}_{len(env or {})}"
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _write_task(root: Path, target_env: dict[str, str] | None = None) -> Path:
    (root / "source").mkdir()
    (root / "target").mkdir()
    (root / "src" / "torchport").mkdir(parents=True)
    task_path = root / "task.json"
    task_path.write_text(
        json.dumps(
            {
                "source": {"repo": "source", "python": "local-source-python"},
                "target": {
                    "repo": "target",
                    "python": "local-target-python",
                    "env": {
                        "KEEP": "yes",
                        "LD_PRELOAD": "stale",
                        "python_config_path": "stale",
                        **(target_env or {}),
                    },
                },
                "inputs": [],
            }
        ),
        encoding="utf-8",
    )
    return task_path


def _bundled_task(
    runner: Any, root: Path, target_env: dict[str, str] | None = None
) -> dict[str, Any]:
    task_path = _write_task(root, target_env)
    base = runner._remote_base(task_path)
    archive = runner._build_bundle(task_path, base)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
        task_file = package.extractfile("task.json")
        assert task_file is not None
        return json.load(task_file)


def test_remote_runner_defaults_are_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)

    task = _bundled_task(runner, tmp_path)

    assert runner.REMOTE_ROOT == PurePosixPath("/root/autodl-tmp/tp-dsh-web")
    assert task["source"]["python"] == "/root/miniconda3/bin/python"
    assert task["target"]["python"] == "/root/autodl-tmp/jt-env/bin/python"
    assert task["target"]["env"] == {
        "KEEP": "yes",
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        "PATH": (
            "/usr/local/cuda/bin:/root/autodl-tmp/jt-env/bin:/usr/local/sbin:"
            "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "python_config_path": "/root/miniconda3/bin/python3.12-config",
        "use_cutlass": "0",
    }


def test_remote_runner_uses_environment_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner(
        monkeypatch,
        tmp_path,
        {
            "TP_REMOTE_ROOT": "/srv/torchport",
            "TP_REMOTE_SOURCE_PYTHON": "/opt/source/bin/python",
            "TP_REMOTE_TARGET_PYTHON": "/opt/target/bin/python",
            "TP_REMOTE_PYTHON_CONFIG": "/opt/source/bin/python-config",
            "TP_REMOTE_LIBSTDCXX": "/opt/lib/libstdc++.so.6",
            "TP_REMOTE_TARGET_PATH": "/cuda/bin:/opt/target/bin:/usr/bin",
        },
    )
    monkeypatch.chdir(tmp_path)

    task = _bundled_task(runner, tmp_path)

    assert runner.REMOTE_ROOT == PurePosixPath("/srv/torchport")
    assert task["source"]["repo"].startswith("/srv/torchport/")
    assert task["source"]["python"] == "/opt/source/bin/python"
    assert task["target"]["python"] == "/opt/target/bin/python"
    assert task["target"]["env"] == {
        "KEEP": "yes",
        "LD_PRELOAD": "/opt/lib/libstdc++.so.6",
        "PATH": "/cuda/bin:/opt/target/bin:/usr/bin",
        "python_config_path": "/opt/source/bin/python-config",
        "use_cutlass": "0",
    }


def test_empty_optional_overrides_are_omitted_from_bundled_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner(
        monkeypatch,
        tmp_path,
        {"TP_REMOTE_LIBSTDCXX": "", "TP_REMOTE_PYTHON_CONFIG": ""},
    )
    monkeypatch.chdir(tmp_path)

    task = _bundled_task(runner, tmp_path)

    assert "LD_PRELOAD" not in task["target"]["env"]
    assert "python_config_path" not in task["target"]["env"]
    assert task["target"]["env"]["KEEP"] == "yes"


def test_remote_settings_can_be_loaded_from_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner(
        monkeypatch,
        tmp_path,
        {"TP_REMOTE_SOURCE_PYTHON": "/environment/python"},
        (
            "TP_REMOTE_ROOT='/from-file'\n"
            'export TP_REMOTE_SOURCE_PYTHON="/file/python"\n'
            "TP_REMOTE_USE_CUTT=0\n"
            "TP_REMOTE_USE_MKL=0\n"
            "TP_REMOTE_NVCC_PATH=/from-file/nvcc\n"
        ),
    )

    assert runner.REMOTE_ROOT == PurePosixPath("/from-file")
    assert runner.REMOTE_SOURCE_PYTHON == "/environment/python"
    assert runner.REMOTE_TARGET_ENV_DEFAULTS == {
        "use_cutt": "0",
        "use_mkl": "0",
        "nvcc_path": "/from-file/nvcc",
    }


def test_remote_target_env_defaults_fill_missing_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner(
        monkeypatch,
        tmp_path,
        {
            "TP_REMOTE_USE_CUTT": "0",
            "TP_REMOTE_USE_MKL": "0",
            "TP_REMOTE_NVCC_PATH": "/usr/local/cuda-11.3/bin/nvcc",
        },
    )
    monkeypatch.chdir(tmp_path)

    task = _bundled_task(runner, tmp_path)

    assert task["target"]["env"]["use_cutt"] == "0"
    assert task["target"]["env"]["use_mkl"] == "0"
    assert task["target"]["env"]["nvcc_path"] == "/usr/local/cuda-11.3/bin/nvcc"


def test_remote_target_env_defaults_do_not_override_task_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner(
        monkeypatch,
        tmp_path,
        {
            "TP_REMOTE_USE_CUTT": "0",
            "TP_REMOTE_USE_MKL": "0",
            "TP_REMOTE_NVCC_PATH": "/configured/nvcc",
        },
    )
    monkeypatch.chdir(tmp_path)

    task = _bundled_task(
        runner,
        tmp_path,
        {
            "use_cutt": "1",
            "use_mkl": "1",
            "nvcc_path": "/task/nvcc",
        },
    )

    assert task["target"]["env"]["use_cutt"] == "1"
    assert task["target"]["env"]["use_mkl"] == "1"
    assert task["target"]["env"]["nvcc_path"] == "/task/nvcc"


def test_empty_remote_target_env_defaults_are_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner(
        monkeypatch,
        tmp_path,
        {
            "TP_REMOTE_USE_CUTT": "",
            "TP_REMOTE_USE_MKL": "",
            "TP_REMOTE_NVCC_PATH": "",
        },
    )
    monkeypatch.chdir(tmp_path)

    task = _bundled_task(runner, tmp_path)

    assert "use_cutt" not in task["target"]["env"]
    assert "use_mkl" not in task["target"]["env"]
    assert "nvcc_path" not in task["target"]["env"]


def test_weights_are_referenced_from_remote_cache_not_bundled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner(monkeypatch, tmp_path, {"TP_REMOTE_ROOT": "/remote/root"})
    monkeypatch.chdir(tmp_path)
    task_path = _write_task(tmp_path)
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"weights-content")
    raw = json.loads(task_path.read_text("utf-8"))
    raw["weights"] = "model.pth"
    task_path.write_text(json.dumps(raw), encoding="utf-8")

    archive = runner._build_bundle(task_path, runner._remote_base(task_path))
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
        names = package.getnames()
        task_file = package.extractfile("task.json")
        assert task_file is not None
        task = json.load(task_file)

    import hashlib

    sha = hashlib.sha256(b"weights-content").hexdigest()
    assert task["weights"] == f"/remote/root/weights-cache/{sha}/model.pth"
    assert not any(name.startswith("weights/") or name.endswith("model.pth") for name in names)


def test_ensure_weights_skips_upload_on_cache_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner(monkeypatch, tmp_path, {"TP_REMOTE_ROOT": "/remote/root"})
    commands: list[str] = []
    monkeypatch.setattr(
        runner.transport,
        "run",
        lambda client, command, timeout=0: (commands.append(command), (0, "", ""))[1],
    )

    class NoSftp:
        def open_sftp(self) -> None:
            raise AssertionError("cache hit must not upload")

    runner._ensure_weights(NoSftp(), tmp_path / "model.pth", "abc")
    assert commands == ["test -f /remote/root/weights-cache/abc/model.pth"]


def test_extra_roots_allow_workspace_outside_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outside = tmp_path / "runs"
    outside.mkdir()
    monkeypatch.setenv("TP_EXTRA_ROOTS", str(outside))
    repo = tmp_path / "repo"
    repo.mkdir()
    runner = _load_runner(monkeypatch, repo)
    # _load_runner 把 TP_PROJECT_ROOT 设为 repo；工作区在 repo 之外但在额外根下
    assert runner._inside_workspace(outside / "ws" / "submission.py") == (outside / "ws" / "submission.py").resolve()
    with pytest.raises(PermissionError):
        runner._inside_workspace(tmp_path / "elsewhere" / "x.py")


def test_extra_roots_default_to_repo_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TP_EXTRA_ROOTS", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    runner = _load_runner(monkeypatch, repo)
    assert runner.EXTRA_ROOTS == ()
    with pytest.raises(PermissionError):
        runner._inside_workspace(tmp_path / "runs" / "x.py")


def _v2_report() -> dict[str, Any]:
    return {
        "kind": "torchport.verify",
        "trial_id": "t0",
        "passed": False,
        "failure_code": "PARITY_FAILED",
        "summary": "PARITY_FAILED: 首个超出噪声包络 model.decoder.MSA5.F_TA.norm1.body",
        "e1": {
            "first_divergence": {"name": "model.decoder.MSA5.F_TA.norm1.body", "position": 896},
            "tensors": [{"name": "model.decoder.MSA5.F_TA.norm1.body", "ratio": 10.7}],
            "metrics": {"compared_tensors": 860, "failed_tensors": 244, "max_ratio": 78.2},
        },
        "results": [
            {
                "runnable": "pass",
                "structural": "pass",
                "numerical": "fail",
                "problems": ["首个超出噪声包络 model.decoder.MSA5.F_TA.norm1.body"],
                "evidence": {"weights": {"complete": True}},
            }
        ],
    }


def test_diagnostic_view_has_no_location(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _load_runner(monkeypatch, tmp_path)
    view = runner.diagnostic_view(_v2_report())
    assert view == {
        "passed": False,
        # v2 报告没有三档状态
        "e1_status": None,
        "stage": "numerical",
        "failure_code": "PARITY_FAILED",
        "structural": "pass",
        "weight_load": "pass",
        "compared_tensors": 860,
        "failed_tensors": 244,
        "max_ratio": 78.2,
    }
    assert "MSA5" not in json.dumps(view)


def test_diagnostic_filter_replaces_full_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _load_runner(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "FEEDBACK", "diagnostic")
    out = runner._filter("verify", json.dumps(_v2_report()))
    assert "MSA5" not in out and "first_divergence" not in out
    monkeypatch.setattr(runner, "FEEDBACK", "localized")
    assert "MSA5" in runner._filter("verify", json.dumps(_v2_report()))


@pytest.mark.parametrize(
    "arguments",
    [
        ["trace", "task.json"],
        ["localize", "/remote/root/x/evidence/trace/t0.json"],
        ["run", "task.json", "target", "--mode", "trace"],
    ],
)
def test_diagnostic_blocks_localizing_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], arguments: list[str]
) -> None:
    runner = _load_runner(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "FEEDBACK", "diagnostic")
    monkeypatch.setattr(runner, "connect", lambda: (_ for _ in ()).throw(AssertionError("must not connect")))
    assert runner.main(arguments) == 2
    assert "不可用" in capsys.readouterr().out
