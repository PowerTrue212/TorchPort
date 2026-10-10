# {{REPO_NAME}}：从 PyTorch 迁移到原生 Jittor

你在一个独立工作区中工作。当前目录是 {{REPO_NAME}} 官方 PyTorch 推理代码的副本，外加一个空的 Jittor 候选骨架。
任务：把该模型的推理实现迁移为原生 Jittor 实现，使其对同一输入、同一官方权重给出与 PyTorch 数值一致的输出。

## 硬性要求

1. 目标是原生 Jittor 重写，不是 jtorch 或任何 PyTorch 兼容层，不得在候选中 import torch、不得用 PyTorch 代跑、
   不得硬编码输入输出、不得放宽误差标准。
2. Jittor 模块的参数名必须与 PyTorch `state_dict` 的键完全一致（含嵌套前缀、buffer 名）。权重由评测方加载，
   你的实现必须无需任何修改即可接收官方 state_dict。参数形状、数量也必须一致，多参数、少参数都算失败。
3. 实现候选入口接口：
   - 在 `submission.py` 中实现 `load_model() -> jt.Module`。
   - 返回的模块 `__call__` 接收一个 NCHW float32 图像 `{{INPUT_SHAPE}}`，返回元组 `tuple[jt.Var, ...]`，
     其结构、顺序、形状必须与 PyTorch 侧 `torchport_adapter.py` 的返回一致。
   - 根目录的 `torchport_adapter.py` 已是稳定入口，不要修改它。
   - {{ADAPTER_NOTE}}
4. 不要下载任何东西（权重、数据、包、代码）。不要联网检索。除下方“工具”一节列出的远端执行脚本和远端工作目录外，
   不要访问工作区之外的任何路径。
5. 只修改 `submission.py` 以及你新增的 Jittor 模块文件。不要修改 PyTorch 原始源码文件，不要修改 `torchport_adapter.py`。
6. 本次只迁移推理路径（eval，batch 推理，fp32）。训练、数据加载、损失函数、评测脚本不需要迁移。

## 访问边界

访问规则分为三级：

1. 以下行为会使本次运行无效：绕过已提供工具，直接导入或执行 TorchPort 库；直接读取 oracle 或 evidence 内容、
   远端 verify 报告 JSON 或 envelope 数据；读取其他工作区或 evaluator。
2. 以下行为禁止，并会记录为实验偏差：读取 TorchPort 源码（`src/torchport/`，以及除调用 `ssh_exec.py` 之外的
   `scripts/remote/`）、仓库文档或 `AGENTS.md`。
3. 允许访问：当前工作区内的文件、已提供的工具、你自己的远端工作目录，以及你在那里自行建立的
   PyTorch 与 Jittor 对照测试。

## 预算与停止规则

- Token 预算：{{BUDGET_TOKENS}}
- 最大对话轮数：{{MAX_TURNS}}
- 最长墙钟时间：{{WALL_MINUTES}} 分钟
- 以下任一情况出现即停止：预算耗尽、轮数用尽、时间用尽、你确认存在无法在当前环境内解决的阻塞，或你认为已完成。
- 不要在结束前反复请求用户确认，自主工作直到停止。
- `verify` 报告中的 `e1_status` 是数值验收的三档结论：`pass`（正式通过）、`marginal`（未正式通过，
  但已达到可提交底线，最终以任务指标为准）、`fail`（任务还没有完成）。只有 `e1_status` 为 `pass` 或
  `marginal` 时才可以结束并提交；为 `fail` 时继续工作，直到预算、轮数或时间用尽。
- `max_ratio` 是所有张量、所有重复运行中 `rel(T,R) / max(D, N_env)` 的最大值：`D` 是 PyTorch 合法运行之间
  的差异，`N_env` 是当前 Jittor 环境自身的运行间波动下限。`max_ratio <= 2` 为 `pass`，`<= 4` 为 `marginal`。

## 工具

<!-- TOOLS:BEGIN -->
{{TOOLS}}
<!-- TOOLS:END -->

## 结束时的汇报

用中文说明：修改了哪些文件、关键设计依据、你认为的当前状态、仍未解决的问题。不要声称“已通过”，最终判定由评测方完成。
