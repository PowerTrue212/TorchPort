# LightGCN full-five-r3 运行手册（给 Windows 上的 DSH 监督者）

你只执行下面的命令并报告输出，不做判断、不改代码、不改 TaskSpec 或基数文件、不碰
`workspaces-*` 下的任何文件。任何一步的输出和"期望"不符：**停下，把原始输出贴给用户**，
不要重试、不要换参数。删除文件、推送、改密钥前先问用户。

你的工作目录必须是 `D:\科研\TorchPort`，不能在任何 `workspaces-*` 目录下启动。

## 背景（只读）

- none-five-r1 的 `RUN_FAILED` 是基础设施问题：远端执行器把空的 `python_config`
  替换成默认的 `python3.12-config`，Jittor 编出 `cpython-312` 扩展，Python 3.9 加载失败
  （`assert jit_utils.cc`），Agent 代码从未执行。已修 `experiments/harness/remote_executor.py`
  并加回归测试。none-five-r2 碰巧依赖缓存里一个来源不明的旧 `cpython-39` 扩展才跑通。
- full-five-r3 已准备好：控制目录 `D:\科研\LightGCN-Benchmark-v1\controller\lightgcn-train-full-five-r3`，
  工作区 `D:\科研\LightGCN-Benchmark-v1\workspaces-train-five-r3\full`，
  `python_config` 显式为 target 环境的 `python3.9-config`，Jittor 缓存为全新的
  `jittor-cache-trial-r3-full`。oracle digest `cbbc8f7c…`，与 none-five-r2 相同。

## 步骤

所有命令在 PowerShell、`D:\科研\TorchPort` 下执行。

### 1. 远端空闲与空间

```powershell
.\.venv\Scripts\python.exe scripts\remote\ssh_exec.py run "nvidia-smi --query-gpu=memory.used --format=csv,noheader; pgrep -af torchport || echo no-proc; df -h /root/autodl-tmp | tail -1" 
```

需要先设置 `$env:TP_SSH_ENV_FILE = "$HOME\.tp_ssh2.env"`。
期望：显存约 `0 MiB`、`no-proc`、可用空间 ≥ 10G。

### 2. 本地前提

```powershell
$env:DEEPSEEK_API_KEY.Length -gt 0
wsl -d Ubuntu -- docker ps --filter name=tp- --format "{{.Names}}"
git -C "D:\科研\LightGCN-Benchmark-v1\workspaces-train-five-r3\full" status --porcelain
```

期望：`True`；docker 无输出（没有残留 trial 容器）；`git status` 无输出（工作区干净）。
代理 `http://127.0.0.1:7890` 需可用。

### 3. 启动 full-five-r3（最长约 3 小时，期间不要并行跑任何 GPU 任务）

```powershell
.\.venv\Scripts\python.exe experiments\harness\agents\run_container_trial.py `
  --controller "D:\科研\LightGCN-Benchmark-v1\controller\lightgcn-train-full-five-r3" `
  --prompt experiments\p2\lightgcn-v1\prompt-train-five.md `
  --proxy http://127.0.0.1:7890 --memory 5g
```

期望：命令正常退出；控制目录出现 `container-runtime.json`、`audit.json`、
`trial\result.json`、`trial\official-verify.json`。

### 4. 确认这次真的用了 Python 3.9 扩展

```powershell
.\.venv\Scripts\python.exe scripts\remote\ssh_exec.py run "find /root/autodl-tmp/LightGCN-Benchmark-v1/jittor-cache-trial-r3-full -name 'jit_utils_core*.so'"
```

期望：只有 `cpython-39` 版本，**没有** `cpython-312`。出现 312 就停下报告。

### 5. 报告原始数字（只填数，不解释）

从控制目录读取并按 [r7 结论表](../simcse-bert-base-v1/train-r7-pilot-outcome.md)
的格式填写：预检和审计命中数、`agent_exit`、模型步数（`audit.json` 的
`model_steps`）、`result.json` 的 `logical_query_cost`（分工具计数、唯一 candidate 数）、
`physical_evaluation_cost`、`agent_wall_seconds`、`stop_reason`、`official-verify.json`
的 `passed`、`failure_code` 和每次运行的超限张量数。

若隐藏验收 stderr 含 `double free`、`malloc(): memory corruption`、`assert jit_utils.cc`、
`fetch failed` 或 MCP 断连，原样标注，不当作 Agent 实现错误。

把数字发给用户；结论文档由用户或 Claude 撰写，不要自己写对比结论。

### 6. 不要做的事

- 不清理远端、不删本地控制目录、不 `git commit` / `git push`。
- 不重跑。如果 trial 失败，只报告原因；是否换 r4 由用户决定。
