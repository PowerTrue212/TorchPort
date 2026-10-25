#!/usr/bin/env python3
"""SSH helper: run a command on a configured remote GPU host, or get/put files.

Usage:
  python scripts/remote/ssh_exec.py run "remote command" [--timeout N] [--workdir DIR]
  python scripts/remote/ssh_exec.py get REMOTE_PATH LOCAL_PATH
  python scripts/remote/ssh_exec.py put LOCAL_PATH REMOTE_PATH
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import paramiko


def load_env_file(path=None):
    """Load the four SSH values without printing or copying them into the repository."""
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
        if name in {"TP_SSH_HOST", "TP_SSH_PORT", "TP_SSH_USER", "TP_SSH_PASSWORD"}:
            os.environ.setdefault(name, value)


# 连接信息只从进程环境或用户目录的 .tp_ssh.env 读取，密码不进代码
def connect():
    load_env_file()
    host = os.environ["TP_SSH_HOST"]
    port = int(os.environ.get("TP_SSH_PORT", "22"))
    user = os.environ.get("TP_SSH_USER", "root")
    password = os.environ["TP_SSH_PASSWORD"]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password,
                   timeout=30, banner_timeout=60, auth_timeout=60)
    return client


def main():
    mode = sys.argv[1]
    client = connect()
    try:
        if mode == "run":
            cmd = sys.argv[2]
            timeout = None
            workdir = None
            i = 3
            while i < len(sys.argv):
                if sys.argv[i] == "--timeout":
                    timeout = float(sys.argv[i + 1])
                    i += 2
                elif sys.argv[i] == "--workdir":
                    workdir = sys.argv[i + 1]
                    i += 2
                else:
                    i += 1
            if workdir:
                cmd = "cd %s && %s" % (workdir, cmd)
            stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
            with ThreadPoolExecutor(max_workers=2) as pool:
                out_future = pool.submit(stdout.read)
                err_future = pool.submit(stderr.read)
                out = out_future.result().decode("utf-8", "replace")
                err = err_future.result().decode("utf-8", "replace")
            code = stdout.channel.recv_exit_status()
            sys.stdout.write(out)
            sys.stderr.write(err)
            sys.stdout.write("\n[remote exit code: %d]\n" % code)
            sys.stdout.flush()
            return code
        elif mode == "get":
            remote, local = sys.argv[2], sys.argv[3]
            sftp = client.open_sftp()
            sftp.get(remote, local)
            sftp.close()
            print("downloaded %s -> %s" % (remote, local))
            return 0
        elif mode == "put":
            local, remote = sys.argv[2], sys.argv[3]
            sftp = client.open_sftp()
            sftp.put(local, remote)
            sftp.close()
            print("uploaded %s -> %s" % (local, remote))
            return 0
        else:
            print("unknown mode", file=sys.stderr)
            return 2
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
