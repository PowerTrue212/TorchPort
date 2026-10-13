# CamoFormer 从 PyTorch 迁移到 Jittor

你正在 TorchPort 项目中工作。请自主检查当前迁移状态，把 CamoFormer 的 PyTorch
推理实现迁移为可以通过 TorchPort 验收的原生 Jittor 实现。

你可以正常使用 DSH 提供的文件浏览、代码搜索、PowerShell、Python、测试和网页搜索，
也可以创建用于排查问题的最小复现。不要等我逐步告诉你该看什么或该怎么修；先阅读项目、
任务说明和当前候选代码，再根据实际运行结果决定下一步。

本次迁移使用这些路径：

```text
PyTorch 源仓库（只读）：CamoFormer/source/HVision-NKU-CamoFormer
Jittor 候选仓库（允许修改）：CamoFormer/outputs/materializer-runtime-package-smoke/workspaces/camoformer-localized-r0/repo
冻结任务：CamoFormer/outputs/materializer-runtime-package-smoke/tasks/camoformer-localized-r0/public.json
实验记录目录：CamoFormer/outputs/dsh-web-camoformer
实验编号：camoformer-dsh-web-r1
```

只修改 Jittor 候选仓库和实验记录目录。不要修改 PyTorch 源仓库、冻结任务、输入数据、
TorchPort 验收代码或已有 oracle，也不要查看或利用隐藏验收数据。候选实现必须真正使用
Jittor，不得用 PyTorch 代跑、硬编码输入输出或放宽误差标准。

需要运行和验收时，使用已经安装到 DSH 的 TorchPort 工具：

- `torchport_inspect`：查看公开任务要求；
- `torchport_run`：运行 PyTorch 源实现或当前 Jittor 候选实现；
- `torchport_evaluate`：比较已经生成的运行产物；
- `torchport_trace`：采集并比较两端的中间结果；
- `torchport_localize`：从 trace 报告提取首次分歧；
- `torchport_verify`：按冻结任务完成最终验收。

调用工具时，把上面的冻结任务路径作为 `task`，把实验记录目录作为 `root`，把实验编号
作为 `trial`。这些 TorchPort 工具会自动把当前代码同步到远端 GPU；普通 PowerShell、
Python 和文件工具仍在本地项目中运行，不需要自行建立 SSH 连接。如果远端工具失败，先按
它返回的原始错误判断是候选代码问题还是运行环境问题；能在候选代码中解决就继续解决，
只有明确属于当前环境的问题时才停止。

请持续工作，直到 `torchport_verify` 明确通过，或者确认存在当前机器无法解决的环境阻塞。
不要把“代码看起来正确”当作通过。结束时用中文说明：修改了哪些文件、关键修复依据、
最后一次验收结果、调用 TorchPort 工具的次数、实际运行耗时、使用过的网络资料链接，
以及仍未解决的问题。
