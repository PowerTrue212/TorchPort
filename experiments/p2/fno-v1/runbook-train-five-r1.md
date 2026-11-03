# FNO 五步训练 trial r1 运行手册（给 Windows 上的 DSH 监督者）

你只执行下面的命令并报告输出，不做判断、不改代码、不改 TaskSpec、提示词或基数文件、不碰
`workspaces-*` 下的任何文件。任何一步的输出和"期望"不符：**停下，把原始输出贴给用户**，
不要重试、不要换参数。删除文件、推送、改密钥前先问用户。

你的工作目录必须是 `D:\科研\TorchPort`，不能在任何 `workspaces-*` 目录下启动。

## 背景（只读）

- 两组都已准备好，**先跑 none，再跑 full，不能同时跑**（共用一张 RTX 4090）。
  - none：控制目录 `D:\科研\FNO-Benchmark-v1\controller\fno-train-none-five-r1`，
    工作区 `D:\科研\FNO-Benchmark-v1\workspaces-train-five-r1\none`，端口 18931，diagnostic。
  - full：控制目录 `D:\科研\FNO-Benchmark-v1\controller\fno-train-full-five-r1`，
    工作区 `D:\科研\FNO-Benchmark-v1\workspaces-train-five-r1\full`，端口 18932，localized。
- 两组同一初始提交 `ae321db`、同一提示词 `experiments\p2\fno-v1\prompt-train-five.md`、
  3 小时墙钟、10 次 oracle query。`python_config` 已显式为 target 的 `python3.9-config`，
  Jittor 缓存分别为 `jittor-cache-trial-r1-none` / `-full`，均已预热。
- 训练任务不支持 trace，localized 组看不到 trace/localize，这是协议设计，不是故障。

## 步骤（每组各做一遍 1–5，先 none 后 full）

所有命令在 PowerShell、`D:\科研\TorchPort` 下执行。先设置：

```powershell
$env:TP_SSH_ENV_FILE = "$HOME\.tp_ssh3.env"
$C = "none"   # 第二遍改成 "full"
$B = "D:\科研\FNO-Benchmark-v1"
```

### 1. 远端空闲与空间

```powershell
.\.venv\Scripts\python.exe scripts\remote\ssh_exec.py run "nvidia-smi --query-gpu=memory.used --format=csv,noheader; pgrep -af 'torchport|run_container' || echo no-proc; df -h /root/autodl-tmp | tail -1"
```

期望：显存约 `1 MiB` 以内、`no-proc`（只命中本命令自身的 `bash -c` 可视为通过）、可用 ≥ 10G。

### 2. 本地前提

```powershell
$env:DEEPSEEK_API_KEY.Length -gt 0
wsl -d Ubuntu -- docker ps --filter name=tp- --format "{{.Names}}"
git -C "$B\workspaces-train-five-r1\$C" status --porcelain
Test-Path "$B\controller\fno-train-$C-five-r1\trial"
```

期望：`True`；docker 无输出；`git status` 无输出；最后一行 `False`（该组从未启动过）。
若 `DEEPSEEK_API_KEY` 不在当前进程，可从 User 作用域注入
（`$env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY","User")`），
不要打印值。代理 `http://127.0.0.1:7890` 需可用。

### 3. 启动 trial（最长约 3 小时，期间不要并行跑任何 GPU 任务）

```powershell
.\.venv\Scripts\python.exe experiments\harness\agents\run_container_trial.py `
  --controller "$B\controller\fno-train-$C-five-r1" `
  --prompt experiments\p2\fno-v1\prompt-train-five.md `
  --proxy http://127.0.0.1:7890 --memory 5g
```

期望：命令正常退出；控制目录出现 `container-runtime.json`、`audit.json`、
`trial\result.json`、`trial\official-verify.json`。

### 4. 确认 Jittor 扩展是 Python 3.9

```powershell
.\.venv\Scripts\python.exe scripts\remote\ssh_exec.py run "find /root/autodl-tmp/FNO-Benchmark-v1/jittor-cache-trial-r1-$C -name 'jit_utils_core*.so'"
```

期望：只有 `cpython-39`，**没有** `cpython-312`。出现 312 就停下报告。

### 5. 报告原始数字（只填数，不解释）

从控制目录读取并按 [LightGCN 结论表](../lightgcn-v1/train-five-trials-outcome-20261102.md)
的格式填写：预检和审计命中数、`agent_exit`、`cleanup_error`（若有）、模型步数（`audit.json`
的 `model_steps`）、`result.json` 的 `logical_query_cost`（分工具计数、唯一 candidate 数）、
`physical_evaluation_cost`、`agent_wall_seconds`、`stop_reason`、`official-verify.json`
的 `passed`、`failure_code`、每次运行的超限张量数和 `max_gating_ratio`；
`trial\events.jsonl` 里每次工具调用的 tool、`response_ok`、`error_type`。

若 stderr 或 `agent.err.log` 含 `double free`、`malloc(): memory corruption`、`assert jit_utils.cc`、
`fetch failed`、`TRANSPORT`、MCP 断连或 WSL/docker 错误，原样标注，不当作 Agent 实现错误。

none 组报告完成后，把 `$C` 改成 `full`，从第 1 步重新开始。

## 不要做的事

- 不清理远端、不删本地控制目录、不 `git commit` / `git push`。
- 不重跑。某组失败只报告原因；是否换 r2 由用户决定。
- 结论和 none/full 对比由用户或 Claude 撰写，你只给数字。
