# LightGCN none-five-r3 运行手册（给 Windows 上的 DSH 监督者）

你只执行下面的命令并报告输出，不做判断、不改代码、不改 TaskSpec 或基数文件、不碰
`workspaces-*` 下的任何文件。任何一步的输出和"期望"不符：**停下，把原始输出贴给用户**，
不要重试、不要换参数。删除文件、推送、改密钥前先问用户。

你的工作目录必须是 `D:\科研\TorchPort`，不能在任何 `workspaces-*` 目录下启动。

## 背景（只读）

- 本次是 full-five-r3 的对照组：同一 seed 提交 `bc1ac846`、同一 oracle digest `cbbc8f7c…`、
  同一 prompt、同一修复后的远端配置（`python_config` 为 target 环境的 `python3.9-config`）。
  唯一差别是反馈层级 `diagnostic`，以及独立的全新 Jittor 缓存 `jittor-cache-trial-r3-none`。
- 控制目录 `D:\科研\LightGCN-Benchmark-v1\controller\lightgcn-train-none-five-r3`，
  工作区 `D:\科研\LightGCN-Benchmark-v1\workspaces-train-five-r3\none`，端口 18833。

## 步骤

所有命令在 PowerShell、`D:\科研\TorchPort` 下执行。先设置：

```powershell
$env:TP_SSH_ENV_FILE = "$HOME\.tp_ssh2.env"
```

`DEEPSEEK_API_KEY` 若当前进程没有，按 full-five-r3 的做法从 User 作用域注入启动进程，
不打印、不落盘。

### 1. 远端空闲与空间

```powershell
.\.venv\Scripts\python.exe scripts\remote\ssh_exec.py run "nvidia-smi --query-gpu=memory.used --format=csv,noheader; pgrep -af 'python.*torchport' || echo no-proc; df -h /root/autodl-tmp | tail -1"
```

期望：显存约 `0 MiB`、`no-proc`、可用空间 ≥ 10G。

### 2. 本地前提

```powershell
$env:DEEPSEEK_API_KEY.Length -gt 0
wsl -d Ubuntu -- docker ps --filter name=tp- --format "{{.Names}}"
git -C "D:\科研\LightGCN-Benchmark-v1\workspaces-train-five-r3\none" status --porcelain
```

期望：`True`；docker 无输出；`git status` 无输出。代理 `http://127.0.0.1:7890` 需可用。

### 3. 启动 none-five-r3（期间不要并行跑任何 GPU 任务）

```powershell
.\.venv\Scripts\python.exe experiments\harness\agents\run_container_trial.py `
  --controller "D:\科研\LightGCN-Benchmark-v1\controller\lightgcn-train-none-five-r3" `
  --prompt experiments\p2\lightgcn-v1\prompt-train-five.md `
  --proxy http://127.0.0.1:7890 --memory 5g
```

期望：命令正常退出；控制目录出现 `container-runtime.json`、`audit.json`、
`trial\result.json`、`trial\official-verify.json`。

### 4. 确认用了 Python 3.9 扩展

```powershell
.\.venv\Scripts\python.exe scripts\remote\ssh_exec.py run "find /root/autodl-tmp/LightGCN-Benchmark-v1/jittor-cache-trial-r3-none -name 'jit_utils_core*.so'"
```

期望：只有 `cpython-39`，**没有** `cpython-312`。

### 5. 报告原始数字（只填数，不解释）

与 full-five-r3 的报告格式相同：预检和审计命中数、`agent_exit`、`model_steps`、
`logical_query_cost`（分工具计数、唯一 candidate 数）、`physical_evaluation_cost`（agent /
official / total）、`agent_wall_seconds`、`stop_reason`、`official-verify.json` 的 `passed`、
`failure_code`、每次运行的超限张量数和 `max_gating_ratio`。另列出 `trial\tool-errors\`
下每个文件的内容，以及 `trial\events.jsonl` 中每次工具调用的 `tool` 和 `response_ok`。

若 stderr 含 `double free`、`malloc(): memory corruption`、`assert jit_utils.cc`、
`fetch failed` 或 MCP 断连，原样标注。结论文档由用户或 Claude 撰写。

### 6. 不要做的事

- 不清理远端、不删本地控制目录、不 `git commit` / `git push`。
- 不重跑。失败只报告原因；是否换 r4 由用户决定。
