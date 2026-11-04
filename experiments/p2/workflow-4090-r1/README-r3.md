# 4090 工作流第三轮（r3）：RAFT none/full 重跑、FNO none 重做

给 Windows 上的 DSH 执行者。规则与红线沿用 `README.md`、`README-r2.md`，这里只写不同之处。
用户可能不在，**不要停下来等回复**：每步把命令、原样输出（最多 40 行）和 PASS/FAIL 追加到
`D:\科研\Workflow-4090-r1\status-r3.md`，失败按本步规定处理后继续。全部结束写一行
`WORKFLOW R3 DONE`。所有结果标 `pilot_only`。

工作目录 `D:\科研\TorchPort`；PowerShell 先设 `$env:TP_SSH_ENV_FILE = "$HOME\.tp_ssh3.env"`。
你写的 `.ps1` 必须存成 UTF-8 with BOM。新脚本放 `experiments\p2\workflow-4090-r1\scripts-r3\`。

## 为什么有 r3（只读）

- r2 的 RAFT none 因共享参数别名（`norm3` / `downsample.1`）被误判 `STRUCTURE_MISMATCH`。
  Harness 已修（提交 `edb09fb`），训练任务另加临界加跑 `marginal_rerun={upper 3.0, runs 5}`。
  用新代码复核：参考两次通过（usage ≈ 0.4），5 个有效负对照全部被拒。
- r2 的 FNO none（r5）接着 r4 中断后的半成品工作区跑，违反"相同初始 candidate"，作废。
- r2 的原判定全部保留，r3 是新一轮，结果不合并进 r2。

## 已由 Claude 完成（只读，不要重做）

| 项 | 位置 |
| --- | --- |
| RAFT none/full 工作区（种子 `2f8bcb5`，与 r2 相同） | `D:\科研\RAFT-Benchmark-v1\workspaces-wf-r3\{none,full}` |
| RAFT 控制目录（端口 18961 / 18962，15 次 query，4 小时） | `D:\科研\RAFT-Benchmark-v1\controller\raft-wf-{none,full}-r3` |
| FNO none 工作区（种子 `3bc3200`，与 full-r1 相同） | `D:\科研\FNO-Benchmark-v1\workspaces-wf-r3\none` |
| FNO none 控制目录（端口 18963，10 次 query，2 小时） | `D:\科研\FNO-Benchmark-v1\controller\fno-wf-none-r6` |
| Jittor 缓存只用控制端参考预热 | 远端 `jittor-cache-wf-r3-*`，日志 `/root/autodl-tmp/workflow-4090-r1/warm-r3.log` |
| 驱动脚本 | `experiments\p2\workflow-4090-r1\scripts-r3\run_trials_wf_r3.ps1` |

训练 oracle digest `5586ca64…` 与 r2 相同，FNO 推理 digest `d0cb4848…` 与 full-r1 相同，不重建 oracle。
提示词不变：RAFT 用 `prompt-raft.md`（两组都含卷积反向缺陷说明），FNO 用 `prompt-fno-inference.md`。

## 阶段 A：开跑前检查

1. 远端：`nvidia-smi --query-gpu=memory.used --format=csv,noheader; pgrep -af 'torchport|run_container' || echo no-proc; df -h /root/autodl-tmp | tail -1`。
   期望显存约 1 MiB、无残留进程、可用 ≥ 15G。
2. `/root/autodl-tmp/workflow-4090-r1/warm-r3.log` 末行 `ALLDONE`，三段 `warm exit=0`、`cpython312=0`。
3. 本地：`DEEPSEEK_API_KEY` 长度 > 0（从 User 作用域注入，不打印值）；`wsl -d Ubuntu -- docker ps --filter name=tp-` 无输出；
   三个工作区 `git status --porcelain` 无输出、`git rev-parse HEAD` 分别是 `2f8bcb5…`、`2f8bcb5…`、`3bc3200…`；
   三个控制目录下**没有** `canaries.json`、`trial\`。
4. 代理：r2 中 `127.0.0.1:7890` 已失效，驱动脚本不传 `--proxy`，三个 trial 同样不传。

任一项不满足：记 FAIL，写明原因，**不启动 trial**，直接写 `WORKFLOW R3 DONE`。

## 阶段 B：运行 trial

用 `Start-Process` 脱离当前会话启动驱动（r2 里随 DSH job runner 被杀导致中断）：

```powershell
Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',`
  'D:\科研\TorchPort\experiments\p2\workflow-4090-r1\scripts-r3\run_trials_wf_r3.ps1' -WindowStyle Hidden
```

驱动按 RAFT none → RAFT full → FNO none 串行跑，每个 trial 前检查远端磁盘（RAFT ≥ 9G、FNO ≥ 3G），
不够就记 `SKIP-DISK` 跳过。进度看 `D:\科研\Workflow-4090-r1\trial-logs-r3\driver-r3.log`，
末行 `R3 DRIVER DONE` 表示结束。每 15 分钟看一次即可，不要干预运行中的 trial。

- **已开始的 trial 一律不重跑**：控制目录里出现过 `canaries.json` 就算开始过，驱动会跳过它。
- Agent 未开始就失败（预检失败、容器没起来）：可新建一个控制目录重试一次，必须同时新建工作区并
  从冻结种子 `git` 复原；不能复用跑过的工作区。写明原因。
- WSL/docker 回收：按 r2 的处理（`wsl -u root -- service docker start`、保活会话），只用于还没开始的 trial。

## 阶段 C：拉回证据并汇总

每个跑完的 trial：

```powershell
.\.venv\Scripts\python.exe experiments\p2\workflow-4090-r1\scripts-r2\pull_all_bases_wf.py `
  <控制目录> /root/autodl-tmp/<RAFT|FNO>-Benchmark-v1/harness-wf-r3 <label>
```

label 用 `raft-none-r3`、`raft-full-r3`、`fno-none-r6`。每个 base 的 `verify.json` 中 `bad` 必须为空。

然后运行 `.\.venv\Scripts\python.exe experiments\p2\workflow-4090-r1\scripts-r3\summarize_trials_r3.py`，
把表格原样贴进状态文件。另外逐个记录：`trial\events.jsonl` 中每次工具调用的 tool、`response_ok`、
`error_type`；`agent.err.log` 中是否出现 `double free`、`cudaErrorIllegalAddress`、`fetch failed`、
MCP 断连或 WSL/docker 错误（原样标注，不当作 Agent 错误）。

## 不要做的事

- 不删除任何远端或本地文件（本轮没有批准任何清理）；磁盘不够就 SKIP。
- 不改 TaskSpec、提示词、基线、`marginal_rerun`、torchport 代码。
- 不 `git commit` / `git push`，不改密钥和代理配置。
- 不写结论和 none/full 对比，只填数字。
