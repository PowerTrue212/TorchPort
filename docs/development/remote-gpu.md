# 远程 GPU 开发

远程主机信息和密码属于本机配置，不写入仓库。`scripts/remote/ssh_exec.py` 从以下
进程环境变量读取连接信息：

```text
TP_SSH_HOST
TP_SSH_PORT
TP_SSH_USER
TP_SSH_PASSWORD
```

可以把这些值保存在仓库外的 `~/.tp_ssh.env`。辅助脚本会自动读取该文件，也可以使用
当前进程中已经设置的同名变量：

```powershell
python scripts/remote/ssh_exec.py run "nvidia-smi"
python scripts/remote/ssh_exec.py put LOCAL_PATH REMOTE_PATH
python scripts/remote/ssh_exec.py get REMOTE_PATH LOCAL_PATH
```

DSH Web 中的六个 TorchPort 工具默认通过 `scripts/remote/dsh_torchport.py` 使用这组
连接信息。每次调用会把当前 candidate、公开 source、输入和 TorchPort 核心同步到远端
专用目录，再执行同名命令。远端 evidence 会跨调用保留，Jittor 缓存仍由 TorchPort 放在
对应 trial 的 cache 目录内。

长期使用时优先在 `~/.ssh/config` 配置主机别名和专用密钥。实验产生的 checkpoint、
编译缓存和输出留在远端数据盘，仓库只保存可复现的任务定义、脱敏报告和必要的小型
证据。
