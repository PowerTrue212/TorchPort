"""Controller-only orchestration of one containerized Agent trial.

plant canaries -> start HTTP MCP -> preflight in container -> DSH in container
-> stop MCP -> audit scan -> certify. All raw outputs stay in the controller dir.

On Windows the docker CLI runs inside a WSL distro (`wsl -d <distro> -- docker ...`);
host paths are translated to /mnt/<drive>/... . On Linux docker is called directly.
The container sees only the candidate at /work, the DSH patch, and a fresh sessions dir.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath
from typing import Any

if __package__ in {None, ""}:
    _project = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_project), str(_project / "src")]

from experiments.harness.agents import audit_trial  # noqa: E402

HERE = Path(__file__).resolve().parent
CONTAINER_WORK = "/work"
CONTAINER_PATCH = "/home/node/trial.patch.yml"
CONTAINER_SESSIONS = "/home/node/.dsh/sessions"


def to_docker_path(path: Path, windows: bool) -> str:
    """Windows 路径转成 WSL 内 docker 可挂载的 /mnt/<盘符>/... 路径。"""
    if not windows:
        return path.as_posix()
    value = PureWindowsPath(path)
    if not value.drive or not value.drive.endswith(":"):
        raise ValueError(f"只支持盘符路径：{path}")
    rest = "/".join(value.parts[1:])
    return f"/mnt/{value.drive[0].lower()}/{rest}"


def docker_prefix(windows: bool, distro: str) -> list[str]:
    return ["wsl.exe", "-d", distro, "--", "docker"] if windows else ["docker"]


def remove_container(docker: list[str], name: str) -> None:
    result = subprocess.run(docker + ["rm", "-f", name], capture_output=True, text=True)
    if result.returncode and "No such container" not in result.stderr:
        raise RuntimeError(result.stderr.strip() or f"无法删除容器：{name}")


def run_args(
    image: str, name: str, workspace: str, mounts: list[tuple[str, str, bool]],
    env_names: list[str], command: list[str], *, interactive: bool = False,
) -> list[str]:
    args = ["run", "--rm", "--name", name, "--network", "host"]
    if interactive:
        args.append("-i")
    args += ["-v", f"{workspace}:{CONTAINER_WORK}", "-v", f"{CONTAINER_WORK}/.scratch"]
    for source, target, read_only in mounts:
        args += ["-v", f"{source}:{target}" + (":ro" if read_only else "")]
    for env in env_names:
        # 密钥只传变量名，值由 docker CLI 从自身环境读取；代理等非密钥可写成 NAME=value
        args += ["-e", env]
    return args + [image, *command]


def proxy_env(proxy: str | None) -> list[str]:
    """容器不继承宿主机代理；显式传入，并让回环地址（MCP）绕过代理。"""
    if not proxy:
        return []
    bypass = "localhost,127.0.0.1,::1"
    return [f"{name}={proxy}" for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")] \
        + [f"{name}={bypass}" for name in ("NO_PROXY", "no_proxy")] \
        + ["NODE_USE_ENV_PROXY=1"]  # Node 24 的内置 fetch 默认不读代理变量


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def start_mcp(config: Path, port: int, token_file: Path, log: Path) -> subprocess.Popen:
    process = subprocess.Popen(
        [sys.executable, str(HERE / "mcp_server.py"), "--trial", str(config),
         "--http", f"127.0.0.1:{port}", "--token-file", str(token_file)],
        stdout=subprocess.PIPE, stderr=log.open("ab"), text=True,
    )
    line = process.stdout.readline() if process.stdout else ""
    if "listening" not in line:
        process.kill()
        raise RuntimeError(f"MCP 服务未启动，详情见 {log}")
    return process


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True,
                        help="prepare_trial.py 生成的控制目录")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--image", default="jittor-harness-agent:0.1")
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--proxy", help="容器出口代理，如 http://127.0.0.1:7890；两组必须相同")
    parser.add_argument("--skip-agent", action="store_true",
                        help="只做 plant/MCP/preflight，用于联调，不启动 DSH 也不 certify")
    args = parser.parse_args()

    windows = os.name == "nt"
    controller = args.controller.resolve()
    trial = json.loads((controller / "container-trial.json").read_text(encoding="utf-8"))
    config_path = controller / "mcp-trial.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidate = Path(config["candidate_root"]).resolve()
    port = int(trial["port"])
    docker = docker_prefix(windows, args.distro)
    name = f"tp-{config['trial_id']}"
    workspace = to_docker_path(candidate, windows)
    runtime: dict[str, Any] = {"started_at": time.time(), "image": args.image,
                               "proxy": args.proxy}
    proxies = proxy_env(args.proxy)

    image_id = subprocess.run(
        docker + ["image", "inspect", "--format", "{{.Id}}", args.image],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    runtime["image_digest"] = image_id
    runtime["docker_version"] = subprocess.run(
        docker + ["version", "--format", "{{.Server.Version}}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    protected = [controller, Path(config["evidence_root"])]
    for path in protected:
        path.mkdir(parents=True, exist_ok=True)
    audit_trial.plant(protected, controller / "canaries.json")

    mcp = start_mcp(config_path, port, controller / "mcp-token.txt", controller / "mcp.err.log")
    agent_exit: int | None = None
    try:
        probe = controller / "probe"
        probe.mkdir(exist_ok=True)
        shutil.copy2(HERE / "preflight.py", probe / "preflight.py")
        shutil.copy2(controller / "preflight.json", probe / "preflight.json")
        remove_container(docker, f"{name}-preflight")
        preflight = subprocess.run(
            docker + run_args(
                args.image, f"{name}-preflight", workspace,
                [(to_docker_path(probe, windows), "/probe", True)], proxies,
                ["python3", "/probe/preflight.py", "--config", "/probe/preflight.json"],
            ),
            capture_output=True, text=True, timeout=600,
        )
        (controller / "preflight-report.json").write_text(preflight.stdout, encoding="utf-8")
        runtime["preflight_exit"] = preflight.returncode
        if preflight.returncode != 0:
            raise RuntimeError(
                f"容器内预检未通过（退出码 {preflight.returncode}），见 preflight-report.json")
        if args.skip_agent:
            return

        sessions = controller / "dsh-sessions"
        sessions.mkdir(exist_ok=False)
        env = dict(os.environ)
        if windows:  # 让 wsl.exe 把 API key 传给 docker CLI
            env["WSLENV"] = ":".join(filter(None, [env.get("WSLENV"), "DEEPSEEK_API_KEY/u"]))
        if not env.get("DEEPSEEK_API_KEY"):
            raise RuntimeError("缺少 DEEPSEEK_API_KEY 环境变量")
        command = docker + run_args(
            args.image, name, workspace,
            [(to_docker_path(controller / "dsh.patch.yml", windows), CONTAINER_PATCH, True),
             (to_docker_path(sessions, windows), CONTAINER_SESSIONS, False)],
            ["DEEPSEEK_API_KEY", *proxies],
            ["dsh", "--profile", "headless", "--patch", CONTAINER_PATCH, "--json", "-"],
            interactive=True,
        )
        remove_container(docker, name)
        runtime["agent_started_at"] = time.time()
        with (args.prompt.open("rb") as stdin,
              (controller / "agent.jsonl").open("wb") as stdout,
              (controller / "agent.err.log").open("wb") as stderr):
            try:
                agent_exit = subprocess.run(
                    command, stdin=stdin, stdout=stdout, stderr=stderr, env=env,
                    timeout=float(config["max_wall_seconds"]) + 600,
                ).returncode
            except subprocess.TimeoutExpired:
                agent_exit = -1
        runtime["agent_finished_at"] = time.time()
        runtime["agent_exit"] = agent_exit
    finally:
        try:
            remove_container(docker, name)
        finally:
            mcp.terminate()
            try:
                mcp.wait(timeout=30)
            except subprocess.TimeoutExpired:
                mcp.kill()
            runtime["finished_at"] = time.time()
            _write_json(controller / "container-runtime.json", runtime)

    rules = json.loads((controller / "audit-rules.json").read_text(encoding="utf-8"))
    report = audit_trial.scan(
        controller / "agent.jsonl", Path(CONTAINER_WORK), controller / "canaries.json",
        rules.get("forbidden_paths", []), rules.get("forbidden_urls", []),
        rules.get("allowed_paths", []), rules.get("max_steps"),
    )
    _write_json(controller / "audit.json", report)
    stop_reason = (
        "scope_violation" if report["scope_violation"]
        else "agent_completed" if agent_exit == 0 else "agent_error"
    )
    from experiments.harness.agents.mcp_server import make_gateway  # noqa: PLC0415

    result = make_gateway(config_path).certify(stop_reason)
    print(json.dumps({
        "stop_reason": stop_reason, "passed": result["passed"], "valid": result["valid"],
        "model_steps": report["model_steps"], "audit_hits": len(report["hits"]),
        "controller": str(controller),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
