# P1/P2 Agent 接口覆盖与效率可比性审计

日期：2026-10-03

## 审计问题

本审计回答两个问题：

1. P1 的 `inspect/run/evaluate/verify/trace/localize` 与 P2 受控 Gateway，是否覆盖了
   CLIP、RAFT、APT 迁移过程中 Agent 实际需要的工具动作？
2. 现有实验记录与 P2 成本字段，是否足以公平比较不同 Agent/反馈条件的迁移效率？

范围包括当前源码、三个实验的冻结协议、prompt、结果、repair rounds、现存 DSH 会话、
APT Codex 原始会话和 P2 tiny smoke。这里的“覆盖”指 Agent 可以在受控接口内完成同类
动作，不要求沿用旧 TorchPort 自动迁移工具的名称。

## 结论

**当前 P1/P2 已覆盖 CLIP、RAFT 推理迁移的主要 Agent 工具面和验证闭环；APT 的长训练、
恢复与异步作业仍不在边界内。现有 CLIP、RAFT、APT 历史结果不能用于正式效率排名。**

- 对 CLIP/RAFT，文件读取、代码修改、受控临时探针、候选执行、数值验证和首次分歧证据
  已经覆盖。依赖安装、远程传输和 worker 配置明确属于实验准备，不暴露给 Agent。
- 对 APT，缺口更大。当前 TaskSpec/capture 固定为构造模型、载入状态、一次前向和张量
  比较，无法表达 10 个任务、每任务 30 epoch、断点恢复和长作业监控。APT 不应直接列入
  当前 P2 正式 benchmark。
- P2 的 Logical Query Cost、Physical Evaluation Cost 和按工具聚合已经能够支持新实验；但
  历史实验没有统一经过 Gateway，因此无法回填 evaluate/verify/trace 次数、物理进程数、
  CUDA 进程秒数和同口径 Time-to-Success。
- repair rounds 只能表示公共 evaluator 失败后的“连贯修复组数”，不能代表工作量。
  RAFT pure DSH 的一轮包含 510 次工具调用，APT Codex 的一轮包含数小时训练；把它们与
  其他方法的一轮直接比较会误导。

## 当前受控接口实际提供什么

P2 Gateway 暴露十一个工具：

```text
list_files  read_file  search_code  apply_patch
inspect     probe      run         evaluate
verify      trace      localize
```

其中：

- `list_files/read_file/search_code/apply_patch` 只作用于冻结 source 或 candidate 工作区；
- `run` 运行 target，`evaluate` 只读最近一次 run；
- `verify` 运行开发协议，trial 结束后由 controller 再执行隐藏 official verify；
- `trace/localize` 只在 localized 条件开放；
- `evaluate/verify/trace` 计 logical query，`run` 只累计实际 physical cost；
- 每次调用、响应 artifact、候选前后身份、持续时间和累计成本写入 `events.jsonl`。

这个边界适合研究“同一个 Agent 在不同验证反馈下的修复效率”。它不是历史 TorchPort
自动迁移器的等价替代；`scan_repository/create_migration_plan/start_migration/
apply_verified_repair` 已由 Agent 的阅读、推理和代码修改承担。

## 工具覆盖矩阵

| Agent 实际动作 | P1/P2 覆盖 | 判断 |
| --- | --- | --- |
| 列目录、读源代码、文本搜索 | 是 | source 只读、target 可读，满足 CLIP/RAFT/APT 代码理解 |
| 创建和修改 candidate 文件 | 是 | `apply_patch(old=null)` 可创建文件，精确替换可修改 |
| 查看冻结任务、输入、权重和协议 | 是 | `inspect` 不泄漏 oracle |
| 运行推理 candidate | 是 | `run` 记录独立 RunManifest 与物理成本 |
| 只读评估现有产物 | 是 | `evaluate` 不重复计算，logical/physical 成本能分开 |
| 重复验证与最终隐藏认证 | 是 | `verify` + controller `certify` |
| 中间 trace 与首次分歧 | 是 | 仅 localized 条件开放，不能签发普通证书 |
| 创建临时诊断脚本后执行 | 是 | `probe` 只执行 candidate 内 `.py`，解释器、环境、设备和超时由 TaskSpec 冻结 |
| 查询框架 API、CUDA、依赖和环境 | 是 | 通过受控 `probe` 查询已准备好的 target 环境；不能安装依赖 |
| 安装依赖、编译扩展 | **否** | 这是实验准备职责；当前也没有显式的 preparation 边界产物 |
| 远程 SSH/SCP | **否** | Gateway 假定运行在实验主机，不负责传输和远程控制 |
| 后台任务、轮询、取消和恢复 | **否** | 当前工具调用同步；没有 job handle |
| 长时间训练、checkpoint/resume | **否** | 当前 capture 强制 `model.eval()` 并执行一次前向 |
| 公共 evaluator repair round | 部分 | 每次工具和 candidate identity 可追踪，但没有“连贯修复组”的显式事件 |
| Agent token/turn/model usage | 是 | adapter 可写 provider 原始 usage；历史实验没有统一写入 |
| 全部工具按名称与耗时聚合 | 是 | aggregate 输出调用数、时间、physical cost、candidate revision 和 milestone |

## CLIP 审计

### 实际过程

CLIP 三种方法都在 round 0 通过，公共 evaluator 时间分别约为：Codex 4.79 秒、pure DSH
4.96 秒、DSH + TorchPort 4.92 秒。这个数字只测最终 Jittor evaluator，不包含阅读代码、
编写 `submission.py`、远程传输和探测环境的时间。

本机保存的两个 DSH 会话显示：

| 方法 | Provider 工具调用 | 主要调用 | Provider 原始 token |
| --- | ---: | --- | --- |
| pure DSH | 44 | `pwsh` 27、write/edit 12、控制类 5 | input 45,233；output 51,588；cache-read 2,129,792 |
| DSH + TorchPort | 107 | `pwsh` 71、read/glob/write/edit 29、旧 TorchPort 工具 3、其他 4 | input 105,506；output 81,410；cache-read 9,059,968 |

DSH + TorchPort 实际只调用过一次 `scan_repository`、一次 `create_migration_plan` 和一次
`generate_certification_report`。迁移、环境探测和公共 evaluator 主要仍通过 `pwsh`
完成。旧自动迁移工具并不是 CLIP 成功所必需的接口；通用代码工具可以替代 scan/plan，
最终报告应由 controller 生成。但没有 probe 时，71 次 shell 动作无法迁入当前 Gateway。

### 是否可比较效率

不能正式比较。`outputs/summary.json` 只保存 rounds 和 evaluator elapsed；原始 DSH 会话
可事后恢复工具与 token，但没有进入冻结结果，Codex 也没有同结构成本记录。round 0
只表示第一次公共 evaluator 就通过，不表示迁移不消耗工作。

## RAFT 审计

### 实际过程

公共结果为：Codex round 5 成功、pure DSH round 1 后停止且 NA=false、DSH + TorchPort
round 1 成功。RAFT 需要大量跨框架局部探针来处理权重 alias、非确定性 reduction、
fusion、grid sampling、BatchNorm 顺序和 CUDA kernel 行为。

现存 DSH 会话中的调用分布为：

| 方法 | 工具调用 | 文件读写搜索 | shell/作业 | TorchPort 结构化工具 |
| --- | ---: | ---: | ---: | ---: |
| pure DSH | 510 | 251 | `pwsh` 231 + `job_output` 21 | 0（但会话错误地暴露了工具） |
| DSH + TorchPort | 517 | 218 | `pwsh` 293 | `generate_certification_report` 2 |

DSH + TorchPort 的 293 次 `pwsh` 中，至少 125 次包含 probe/debug/compare 类动作，6 次包含
公共 evaluator。大多数 TorchPort trace 也通过远程 shell 调用，而不是结构化工具。因此
这些历史动作现在可由受控 `probe` 表达，但只有重新经过 Gateway 的新 trial 才有同口径
成本；历史会话不能据此回填。

纯 DSH 已导出的 `agent_metrics.json` 记录 510 次工具调用、468 次模型调用、14,460.223 秒
session wall 和 provider token；Codex 与 DSH + TorchPort 的成本在正式结果中是
`not_recorded`。原始 DSH + TorchPort 会话虽可恢复 517 次工具调用和 provider usage，仍
缺少同口径 physical executions、CUDA 进程秒数与首次成功时间。

此外，pure DSH 的终止记录明确说明该会话运行时仍暴露 TorchPort 工具，尽管没有调用；
它不是严格隔离的 baseline。RAFT 现有结果可以说明结果与修复轨迹，不能形成正式效率
排序。

## APT 审计

### 实际过程

APT 不是推理 task。它要求 10 个 class-incremental tasks、每个 30 epoch、完整训练、
checkpoint/progress 恢复和最终 10,000 张图推理。当前只有 Codex 的冻结 V100 结果和原始
会话完整；APT DSH 会话、工作区和 evaluator 结果未观测到，DSH + TorchPort 也没有可用
结果。现有报告已经把三方比较标记为不成立。

Codex 原始会话含 430 个 exec tool envelope 和 2 个 wait；进一步展开可见 206 次命令
执行、214 次长任务输入/轮询和 11 次补丁。185 次命令涉及远程 SSH/SCP，77 次涉及训练
或任务监控，67 次涉及环境/依赖检查。round 0 的完整训练耗时 13,673.215 秒，round 1
恢复评估约 48.58 秒；会话墙钟约 9 小时 5 分。

这些动作不能由当前同步、推理专用的 Gateway 表达。即便增加一个 probe，也不能可靠
管理多小时训练、checkpoint、断线恢复和异步状态。

### 是否可比较效率

不能。DSH 证据缺失，人工迁移没有会话日志，Codex 与人工结果又使用不同 GPU、Jittor
版本、seed、数据增强和 evaluator。当前 APT 只能作为“训练迁移需要哪些接口”的需求
样本，不能进入 P2 正式效果/效率比较。

## 效率指标的可用性

| 指标 | 历史 CLIP | 历史 RAFT | 历史 APT | 新 P2 trial |
| --- | --- | --- | --- | --- |
| 最终成功/RN/NA | 可用 | 可用 | 仅 Codex 可用 | 可用 |
| Repair rounds | 可用但很粗 | 可用但很粗 | 仅 Codex | 可由事件派生 |
| Agent wall time | 未冻结 | 仅 pure DSH 正式保存 | 仅 Codex 可重建 | 可用 |
| 全部 tool calls | DSH 会话可重建 | DSH 会话可重建 | 仅 Codex 可重建 | 原始事件可用 |
| Token | DSH 会话可重建 | DSH 会话可重建 | 仅 Codex可重建 | 可用 |
| evaluate/verify/trace query | 不可回填 | 不可可靠回填 | 不适用 | 可用 |
| physical executions/wall/CUDA seconds | 不可回填 | 不可回填 | 不可回填 | 可用 |
| Time-to-Success | 不可同口径回填 | 不可同口径回填 | 不可比较 | 可用 |
| 稳态推理性能 | 最终 evaluator 时间不是该指标 | 同左 | 训练任务不适用 | 通过后可用 |

P2 能比较效率的前提是所有条件从同一初始 candidate、同一 Gateway、同一 provider/model、
同一预算和同一 official evaluator 重新运行。历史结果只能作为任务设计与难度证据，不能
与新的 P2 trial 混在一个效率表中。

## 最小修正建议

> 2026-10-03 收尾状态：受控 `probe`、按工具调用/时间/physical cost 聚合、candidate
> revision 与首次 run/开发 verify milestone 已实现；probe 的越界、参数、凭证继承、超时和
> 输出上限均有测试。下文“否/缺口”保留的是审计当时的历史结论。CLIP/RAFT 已增加 P2
> TaskSpec 物化器；当前 GPU 主机没有 container/bubblewrap，因此本轮只能标为
> `trusted_candidate_pilot`，不能宣称 hidden oracle 获得了强宿主隔离。

### 1. 为 CLIP/RAFT 增加一个受控 `probe`

不要恢复通用 shell，也不要恢复自动迁移器。增加一个受控工具即可覆盖最大的真实缺口：

```text
probe(path, args=[])
```

约束建议：

- `path` 必须位于 target workspace，且由 Agent 先通过 `apply_patch` 创建；
- 只允许使用 TaskSpec 冻结的 target Python，参数按 argv 传递，不经 shell；
- cwd、环境、device、timeout、stdout/stderr 和 candidate identity 全部记录；其中解释器、
  device 和 timeout 取自冻结 TaskSpec，不能由 Agent 覆盖；
- 禁止读取 hidden task/oracle；网络默认关闭；
- 每次实际进程计入 Physical Evaluation Cost，但不计 oracle query；
- probe 输出不能签发成功，也不能替代 `evaluate/verify`。

这样可以表达 RAFT 的最小算子实验、框架 API 探测和 candidate 自测，同时仍保持受控边界。

### 2. 不为 APT 扩展当前 P1/P2

TorchPort 2.0 第一版已限定推理。为了一个尚无三方证据的训练 case 加入异步 job、训练
checkpoint、恢复协议和训练指标，会显著扩大核心。APT 保留为未来 training harness 的
需求样本；当前正式 benchmark 先使用 CLIP、RAFT 和推理型真实任务。

### 3. 补齐聚合字段

不增加新的成本概念，只把现有事件确定性聚合出来：

- `tool_calls_by_name` 与 `tool_time_seconds_by_name`；
- 首次 `run` 成功、首次开发 `verify` 通过和最终 official verify 的时间点；
- candidate revision 数与被查询的 unique candidate 数；
- probe 的 CPU/CUDA 进程数和进程墙钟。

保留现有 Logical Query Cost 与 Physical Evaluation Cost 的分离。文件读取和 patch 次数
可以辅助解释工作量，但不能与 oracle query 合并。

上述字段现由 `experiments.harness.aggregate` 从追加事件确定性计算，不增加第二套计费
状态。每个 tool 的 physical cost 使用相邻事件累计值的差值，official evaluator 仍单列。

### 4. 重新定义历史实验的用途

- CLIP：验证简单模型迁移和 first-pass ceiling，不承担效率结论。
- RAFT：用于验证复杂数值修复、trace 价值和 probe 覆盖；重新跑 P2 trial 后才能比较效率。
- APT：作为训练迁移接口需求与 evaluator 风险案例，不进入当前推理 benchmark 排名。

## 审计后的准入条件

CLIP/RAFT 正式进入 P2 前至少满足：

1. 使用当前 TaskSpec 和隐藏 official evaluator 重建任务；
2. 同一 Agent/provider/model、初始 candidate、预算和硬件；
3. 所有条件只通过 Gateway 操作，禁止旁路 shell；
4. `probe` 进程和 run/verify/trace 进程全部计入 physical cost；
5. 预注册 verdict/diagnostic/localized 条件与 replicates；
6. 失败、预算耗尽和 infrastructure error 全部保留；
7. 历史结果只作为外部参照，不并入 Success@Budget 或成本聚合。

满足这些条件后，P2 可以比较“同一个 Agent 获得不同 TorchPort 反馈时，成功率和成本如何
变化”。在此之前，最多只能比较最终正确性与案例轨迹，不能声称 TorchPort 提高了迁移
效率。
