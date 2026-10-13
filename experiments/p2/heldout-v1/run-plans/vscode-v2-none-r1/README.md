# VSCode-v2-T none-r1：diagnostic 单条件试跑

这是一次新的 v3 试跑，不复用历史 `vscode-v2-full-r0` 候选。目录名沿用 `none`，实际反馈层级是
`diagnostic`：DSH 只开放 inspect、run(normal)、evaluate、verify，不返回任何模块、文件、行号、算子、
张量名或首次分歧信息。

## 已准备的区域

- Agent 工作区：`D:\tp-runs\vscode-v2-none-r1`
- 隐藏 task：`experiments\p2\heldout-v1\workspaces\_tasks\vscode-v2-none-r1.json`
- 本地审计目录：`D:\tp-runs\_control\vscode-v2-none-r1\none`
- 远端 Agent 目录：`/root/autodl-tmp/agent/vscode-v2-none-r1`
- 远端 TorchPort task 根：`/root/autodl-tmp/tp-work/d83817e76bcb`

冻结输入和 `swinT.pth` 已用共享缓存硬链接到远端 `staged/`。不要把 task、TorchPort evidence、
评测结果或 control 目录复制进 Agent 工作区。

准备时远端磁盘剩余 7.5 GB。历史 VSCode-v2 full-r0 的完整 task 根约 4.2 GB，因此够跑一轮，
但不适合同时再开另一轮大型 verify。运行中若剩余空间低于 2 GB，应先停止新增调用并检查。

## 启动 none

在一个新的 `cmd.exe` 窗口中运行以下命令。项目 Agent 不代替用户启动 `dsh web`。

```bat
set TP_PROJECT_ROOT=D:\科研\TorchPort
set TP_EXTRA_ROOTS=D:\tp-runs
set TP_FEEDBACK=diagnostic
set TP_AGENT_REMOTE_DIR=/root/autodl-tmp/agent/vscode-v2-none-r1
set TP_AGENT_AUDIT=D:\tp-runs\_control\vscode-v2-none-r1\none\agent_ssh.jsonl
set TORCHPORT_PYTHON=D:\科研\TorchPort\.venv\Scripts\python.exe
set TORCHPORT_PYTHONPATH=D:\科研\TorchPort\src
set TORCHPORT_RUNNER=D:\科研\TorchPort\scripts\remote\dsh_torchport.py
set TORCHPORT_TRACE_DIR=D:\tp-runs\_control\vscode-v2-none-r1\none\dsh
cd /d D:\tp-runs\vscode-v2-none-r1
dsh web
```

首条消息使用工作区中的 `PROMPT.md`。TorchPort 工具参数为：

- `task`：`D:\科研\TorchPort\experiments\p2\heldout-v1\workspaces\_tasks\vscode-v2-none-r1.json`
- `root`：`evidence`
- `trial`：`vscode-v2-none-r1`

启动后先核对工具列表只有四项，而且 `torchport_run` 只有 normal 模式。若出现 trace、localize 或
trace mode，应立即停止，本 attempt 记为基础设施错误。

## 会话结束后

先停止 DSH 会话，保存会话 ID、实际模型标识、步数、时间、Token、费用和停止原因，再检查
`agent_ssh.jsonl`。不要把隐藏验收的完整报告发回 DSH。

独立终端中的 E1 命令：

```bat
D:\科研\TorchPort\.venv\Scripts\python.exe D:\科研\TorchPort\experiments\p2\heldout-v1\evaluator\run_official_verify.py ^
  D:\科研\TorchPort\experiments\p2\heldout-v1\workspaces\_tasks\vscode-v2-none-r1.json ^
  D:\科研\TorchPort\experiments\p2\heldout-v1\results\vscode-v2-none-r1\official-verify.json ^
  --trial vscode-v2-none-r1-official --session-ended
```

单独这一轮只能得到一个试跑结果，不能据此估计 none 条件的迁移成功率。
