# DSH 实验提示

你正在 TorchPort 项目中工作。请使用 DSH 自带的文件浏览、代码搜索、PowerShell、
Python、测试和网页搜索能力，自主完成指定 PyTorch 仓库到 Jittor 的迁移。

开始后先检查仓库、任务文件和当前候选实现，自己判断当前状态与下一步，不要假定提示词
已经给出了修复方案。可以修改候选迁移仓库和本次实验输出；不要修改冻结任务、隐藏输入、
TorchPort 的验收规则或源仓库。

迁移过程中使用已经安装的 `torchport_inspect`、`torchport_run`、
`torchport_evaluate`、`torchport_trace`、`torchport_localize` 和
`torchport_verify` 运行候选、验收结果和定位首次分歧。每次
修改后根据实际证据决定是否继续读取源码、写最小复现、运行测试或再次验收。不要用
PyTorch 代替 Jittor，不要硬编码输入输出，不要降低容差，也不要编造通过结果。

请持续工作到 TorchPort 验收通过，或者出现无法在当前环境内解决的明确基础设施阻塞。
结束时说明：修改了什么、为什么修改、最终验收是否通过、TorchPort 工具调用次数、运行
耗时，以及仍然存在的限制。

本次任务：

```text
源仓库：<填写路径>
候选仓库：<填写路径>
冻结任务：<填写 task.yaml 或 task.json 路径>
实验编号：<填写稳定编号>
```
