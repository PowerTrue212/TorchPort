# LoRA GPT-2 Small：DSH none / full 试验准备

2026-10-24 r1 试跑的技术验收和实验有效性结论见
[r1 试跑结案](r1-pilot-outcome-20261024.md)。

`none` 是 diagnostic（无定位反馈），`full` 是 localized（有 trace/localize）；两组的
普通 Agent 能力、初始候选和基础 prompt 相同。`materialize.py` 生成候选，
`prepare_trial.py` 将已校准的推理与训练 TaskSpec 绑定到某个候选，并生成控制端 MCP 与
DSH patch 文件。控制文件不能放进 Agent 可见工作区。

## 当前证据与边界

已知正确的人工校准 candidate 在 V100 上分别通过 GPT-2 Small 前向和连续四步训练检验。
训练检验覆盖 loss、可训练参数集合、每步学习率和每步 LoRA 参数；这证明两项验证链路
可运行，不证明空白候选会通过。校准实现与候选隔离，不能作为 Agent 输入。
2026-10-24 在恢复的 V100 上重建训练 oracle，正确 candidate 连续四次 target 均通过；
`loss-sum`、`schedule-shift` 和 `adam-l2` 三个负对照已被相应检查拦下。
原始证据和剩余边界见 [V100 复测记录](train/v100-calibration-20261024.md)。
新 RTX 4090 的前向校准通过，但训练同一 oracle 的四次 target 运行出现两次失败、
两次通过；详见 [4090 校准记录](train/4090-calibration.md)。因此 4090 仍不适合
据此启动正式 trial。

两个任务共享候选仓库与冻结 base+LoRA 权重：`verify` 分别验收，两个都通过才通过；
`run`/`evaluate` 可选 inference 或 training；`trace`/`localize` 只针对 inference。
Harness 在可信控制端执行远端 GPU 命令，然后用同一反馈裁剪器返回 none/full 的差异，
不把远端旧版裁剪器当成第二套判定。MCP 只增加 Harness 工具；DSH 自带的编辑、Shell、
Git、测试、网页搜索与远程代码搜索不关闭。

本轮固定 token batch 足以进行前向和四步训练试跑。E2E 原始 train/valid/test 已在本地，
不需要重新下载；真实任务性能 `P_t` 仍缺编码后的数据、训练过的 GPT-2 Small LoRA
checkpoint、固定的生成与评分流程，以及指标容差。现有 base checkpoint 与合成 LoRA
fixture 不能代替任务训练产物。

## 本地准备

在项目目录之外选一个独立工作区根目录，并把控制文件放在另一处。生成器拒绝覆盖既有目录；
不使用 `--force` 重建已经交给 Agent 的候选。旧 r0 工作区与原始记录保持不变，下面生成新 r1。

```powershell
.\.venv\Scripts\python.exe experiments\p2\lora-gpt2-small-v1\materialize.py --out 'D:\LoRA-Agent-Workspaces\candidates' --replicate 1
.\.venv\Scripts\python.exe experiments\p2\lora-gpt2-small-v1\prepare_trial.py --candidate 'D:\LoRA-Agent-Workspaces\candidates\lora-gpt2-small-full-r1' --controller-dir 'D:\科研\LoRA-Benchmark-v1\controller\full-r1-pilot' --condition full --trial-id lora-full-r1-pilot
```

`controller/full-r0-pilot/dsh.patch.yml` 是 `dsh headless --patch <文件>` 的配置叠加，
将 DSH 的 workspaceRoot 指向候选目录，并添加 stdio MCP。启动命令自身也必须以候选目录
为 `cwd`；DSH 会话记录的 cwd 优先于 patch 中的回退 workspaceRoot。`mcp-trial.json`、任务文件、
MCP 工具返回的原始证据和最终 `trial/` 均留在 Agent 工作区外。MCP 进程重启时会恢复
查询预算、运行清单和事件序号，不能重置预算。长工具调用超时设为 65 分钟。

本轮采用开放工具设置：Agent 保留普通文件、Shell、网页和代码搜索能力。把工作区作为
启动 `cwd`、把控制文件放在另一处、使用中性的候选入口与 prompt，只能减少偶然接触
实验内部材料，不能证明 Agent 无法主动搜索到它们。两组从各自的新会话启动，使用相同
普通工具、基础 prompt、候选 seed 和预算；保存工具调用及公开参考来源，并单独报告
发现人工迁移或内部评测材料的情况。不能把这套布局称为硬隔离，也不能仅凭 prompt
推断所有外部访问都被记录。

Agent 会话结束后，由可信控制端运行
`experiments/harness/agents/certify_trial.py --trial <mcp-trial.json> --stop-reason agent_completed`。
它冻结完整候选仓库，再用两项 task 分别做隐藏验收；不能作为 Agent 工具挂载。
这个 MCP 试跑适配层仍强制 `pilot_only`，即使两项技术验收通过，`valid`/`success`
仍为 false。要把开放工具试跑升级成正式结果，须先把当前要求硬隔离的
`docs/benchmark-v1-protocol.md` 与 run-plan 验证规则改成一致的开放工具协议，并明确
结论只适用于这种设置。

当前 `docs/benchmark-v1-protocol.md` 仍描述更严格的隔离与全指标方案。r1 不以额外人工
训练负对照或完整 E2E 指标作为开跑条件；Agent 在普通工具环境中自行检查训练行为，
隐藏 `verify` 仍给出客观结果。若以后把 r1 计入正式矩阵，须先统一协议与实际设置，
不能把开放工具 pilot 追认为原严格方案的结果。
