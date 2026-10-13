# PlantCamo r1：diagnostic 与 localized 配对试跑

这是一对新的 v3 试跑，不复用历史 `plantcamo-full-r0` 候选。

| 条件目录 | 实际反馈层级 | DSH 可见工具 | 位置信息 |
| --- | --- | --- | --- |
| `plantcamo-none-r1` | `diagnostic` | inspect / run(normal) / evaluate / verify | 不可见 |
| `plantcamo-full-r1` | `localized` | 六个 TorchPort 工具 | 可见 |

`none` 是沿用的目录名；这次实际含义是“没有定位反馈”，不是“没有 TorchPort”。两个工作区的
PyTorch 源码、空 submission、输入、权重、v3 协议和预算相同。Prompt 只在工具段和工作区名上不同。

## 已准备的区域

- Agent 工作区：`D:\tp-runs\plantcamo-none-r1`、`D:\tp-runs\plantcamo-full-r1`
- 隐藏 task：
  - `experiments\p2\heldout-v1\workspaces\_tasks\plantcamo-none-r1.json`
  - `experiments\p2\heldout-v1\workspaces\_tasks\plantcamo-full-r1.json`
- 本地审计目录：`D:\tp-runs\_control\plantcamo-pair-r1\none`、`...\full`
- 远端 Agent 目录：`/root/autodl-tmp/agent/plantcamo-none-r1`、`.../plantcamo-full-r1`

两个远端目录都已预置冻结输入和完整 checkpoint。不要把 `_tasks`、评测结果或 control 目录复制进
Agent 工作区。

启动前先从当前目录版本刷新一次插件，并记录 DSH 实际模型标识；`run-plan.json` 中的 `model_id`
在记录前保持 `null`，不能事后凭印象补写。

```bat
cd /d D:\科研\TorchPort\integrations\dsh
npm test
dsh plugin --profile web add .
```

## 启动 none

在一个新的 `cmd.exe` 窗口中设置以下变量，再由用户运行 `dsh web`：

```bat
set TP_PROJECT_ROOT=D:\科研\TorchPort
set TP_EXTRA_ROOTS=D:\tp-runs
set TP_FEEDBACK=diagnostic
set TP_AGENT_REMOTE_DIR=/root/autodl-tmp/agent/plantcamo-none-r1
set TP_AGENT_AUDIT=D:\tp-runs\_control\plantcamo-pair-r1\none\agent_ssh.jsonl
set TORCHPORT_PYTHON=D:\科研\TorchPort\.venv\Scripts\python.exe
set TORCHPORT_PYTHONPATH=D:\科研\TorchPort\src
set TORCHPORT_RUNNER=D:\科研\TorchPort\scripts\remote\dsh_torchport.py
set TORCHPORT_TRACE_DIR=D:\tp-runs\_control\plantcamo-pair-r1\none\dsh
cd /d D:\tp-runs\plantcamo-none-r1
dsh web
```

首条消息使用工作区里的 `PROMPT.md`。TorchPort 工具调用使用：

- `task`：`D:\科研\TorchPort\experiments\p2\heldout-v1\workspaces\_tasks\plantcamo-none-r1.json`
- `root`：`evidence`
- `trial`：`plantcamo-none-r1`

启动后先核对工具列表只有四项，且 `torchport_run` 只有 normal 模式。如果出现 trace、localize 或
trace mode，立即停止，本 attempt 记为基础设施错误，不让 Agent 继续。

## 启动 full

使用另一个新的 `cmd.exe` 窗口，不能沿用 none 会话：

```bat
set TP_PROJECT_ROOT=D:\科研\TorchPort
set TP_EXTRA_ROOTS=D:\tp-runs
set TP_FEEDBACK=localized
set TP_AGENT_REMOTE_DIR=/root/autodl-tmp/agent/plantcamo-full-r1
set TP_AGENT_AUDIT=D:\tp-runs\_control\plantcamo-pair-r1\full\agent_ssh.jsonl
set TORCHPORT_PYTHON=D:\科研\TorchPort\.venv\Scripts\python.exe
set TORCHPORT_PYTHONPATH=D:\科研\TorchPort\src
set TORCHPORT_RUNNER=D:\科研\TorchPort\scripts\remote\dsh_torchport.py
set TORCHPORT_TRACE_DIR=D:\tp-runs\_control\plantcamo-pair-r1\full\dsh
cd /d D:\tp-runs\plantcamo-full-r1
dsh web
```

工具参数改为 `plantcamo-full-r1.json` 和 trial `plantcamo-full-r1`。启动后应看到六个 TorchPort 工具。

## 会话结束与隐藏验收

1. 先停止该 DSH 会话，保存会话 ID、模型标识、步数、开始/结束时间、Token 原始分类和费用。
2. 检查对应的 `agent_ssh.jsonl`，确认没有越界命令；保留完整日志。
3. 复制 `result-template.json` 为条件对应的结果记录，填写 DSH 会话事实。
4. 在独立终端运行同一份 v3 task 的 official verify；完整 JSON 只能写入评测方结果目录。
5. 不把 official verify 的逐张量结果、`first_divergence`、`first_marginal`、模块构造参数差异或
   自由文本问题描述粘贴回 DSH。只在实验汇总中记录最终 `passed/e1_status`。

这一对只能得到 `0/1` 或 `1/1` 的试跑结果，不能单独估计稳定的迁移成功率。成功率需要用同一模型、
同一预算和干净 seed 继续增加预注册 replicate；后续 replicate 应交替或随机安排条件顺序。

隐藏验收使用专门脚本。它要求显式确认会话已结束，只允许把完整报告写入
`experiments\p2\heldout-v1\results\`，终端只打印不含位置的三项摘要。例如 none：

```bat
D:\科研\TorchPort\.venv\Scripts\python.exe D:\科研\TorchPort\experiments\p2\heldout-v1\evaluator\run_official_verify.py ^
  D:\科研\TorchPort\experiments\p2\heldout-v1\workspaces\_tasks\plantcamo-none-r1.json ^
  D:\科研\TorchPort\experiments\p2\heldout-v1\results\plantcamo-none-r1\official-verify.json ^
  --trial plantcamo-none-r1-official --session-ended
```

full 使用对应的 full task、结果目录和 trial 名。不要从 DSH 的终端工具调用这个脚本。
