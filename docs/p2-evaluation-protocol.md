# P2：成本与最小性能协议

## 研究问题

P2 只回答一个问题：在相同 Agent、任务、初始 candidate、硬件和预算下，TorchPort
验证反馈的粒度是否提高迁移成功率，并以多少时间、Token 和 oracle query 换取该提升。

比较三种条件：

| condition | Agent 可见的验证反馈 |
| --- | --- |
| `verdict` | `evaluate` / `verify` 只返回是否通过 |
| `diagnostic` | 增加失败阶段、通用类别/失败码和聚合数值指标 |
| `localized` | 再开放 `trace` 和固定的首次分歧定位 |

三组使用同一模型版本、基础 prompt、源仓库修订、初始 candidate、任务说明、预算和
最终 evaluator。后台始终运行同一个 TorchPort evaluator，只裁剪返回给 Agent 的字段。

反馈边界在实验开始前冻结：

| 字段 | `verdict` | `diagnostic` | `localized` |
| --- | --- | --- | --- |
| `passed` | 是 | 是 | 是 |
| `stage/category/failure_code` | 否 | 固定类别枚举 | 是 |
| `max_abs/mismatch_ratio` 与不匹配计数 | 否 | 是 | 是 |
| `structural/weight_load/aligned/strict_aligned` | 否 | 是 | 是 |
| `scope/invocation/operator/first_divergence` | 否 | 否 | 是 |
| 输入/输出证据与 `error_jump` | 否 | 否 | 是 |
| `contract`（接口契约错误，固定枚举文本） | 否 | 是 | 是 |

`diagnostic` 使用代码中的固定 allowlist，不返回自由文本 `summary`、`problems` 或 `next`，
也不返回 module、file、line、operator、张量名称或首次分歧。失败码必须归一到通用类别，
例如 `NUMERICAL_MISMATCH`，不能把定位信息编码进失败原因。

例外是接口契约错误（2026-10-28，feedback schema v2）：Harness 自身的适配代码在读取任务契约
（训练函数、optimizer 的 `param_groups`/`lr`、loss 的形状与类型）时失败，失败码为 `CONTRACT_ERROR`，
`contract` 字段给出 `{code, message}`，`message` 只能取代码中冻结的枚举文本，描述接口约定本身，
不含模块、算子、张量名或数值位置。这类错误由 Harness 抛出，Agent 在本地无法复现，若只返回
`RUN_FAILED` 会让 trial 测的是“猜中接口约定”而不是迁移能力；因此 diagnostic 与 localized 都返回它。

Agent 使用成熟代码环境自带的文件浏览、搜索、编辑、Shell、Python、Git、测试和公开网络检索。
Harness 不重写这些能力，只增加 `inspect`、`run`、`evaluate`、`verify`、`trace`、`localize`
六个 TorchPort 工具。`trace` 和 `localize` 只在 localized 条件出现；trace 目前只支持推理任务，
trial 的任务全是 `train-step` 时这两个工具不向 Agent 暴露（2026-11-02 起，此前 trial 原样保留）。
此时 localized 与 diagnostic 的差别只剩 evaluate/verify 反馈中的首次分歧等定位字段。
trace 只做聚焦采集（2026-11-04 起）：必须先有同一 trial 的 verify 报告且其中有首次分歧或临界位置，
trace 才只记录该位置之前 8 个张量所属的模块。此前没有 verify 报告时会退回全量逐层记录，RAFT 20 次
迭代的全量 trace 单次写出 20 GB 以上，两次把 50 GB 数据盘写满（raft-wf-full-r3/r4，记为基础设施
失败，原记录保留）。现在：trial 内从未调用 verify 时，Gateway 在派发前拒绝 trace（不计 query）；
已 verify 但报告里没有可聚焦的位置（例如已通过）时，trace 不启动任何进程，返回 `TRACE_NO_FOCUS`
（到达验证服务，按下条规则计 query，物理成本为 0）。聚焦采集在写入 NPZ 前按数组原始字节数及容器开销
估算本端 trace 产物；估算超过 4 GiB 时拒绝写入，返回 `TRACE_TOO_LARGE`，不继续运行另一端。
这个上限约束单次采集写入量，不包含已经冻结的 oracle。最终认证入口不注册为
Agent tool，隐藏 TaskSpec、oracle 和 evaluator 也不在 candidate 工作区内。

Agent 在迁移前自行检查依赖、运行设备、显存、内存、磁盘、数据和 checkpoint，并决定是否安装、
下载、调整环境或停止。Harness 只记录外部 supervisor 提交的检查事件，不把这些检查固化成一套
仓库无关的真假判断。Gateway 是反馈和证据边界，不是操作系统沙箱；正式实验仍应在隔离 worker
中运行整个 trial，防止 candidate 主动遍历宿主文件系统、凭据或隐藏评测材料。

## 产物

P2 不建立数据库。每个 trial 使用独立目录并保留五类文件：

```text
experiments/p2/runs/<experiment>/<task>/<condition>/<trial>/
├── trial.json             冻结配置和内容摘要
├── events.jsonl           Agent、工具和 evaluator 的追加事件
├── delivery/
│   ├── repository/        最终候选仓库的冻结快照
│   ├── candidate-manifest.json
│   ├── git-status.txt
│   └── candidate.diff
├── official-verify.json   trial 结束后的独立验收原始结果
├── result.json            从原始事件确定性计算的指标
└── performance.json       仅对正确性通过的 candidate 运行
```

`trial.json` 至少包含：

- `experiment_id`、`task_id`、`condition`、`trial_id`、`replicate`。
- Agent provider、模型、版本或不可变标识、关键采样参数。
- source、initial candidate、TaskSpec、prompt 的内容摘要。
- CPU、GPU、驱动、CUDA、双端 Python 和框架版本。
- wall、turn、Token 预算以及预先定义的停止策略。
- oracle query 预算和反馈条件。

`events.jsonl` 只追加原始事实：时间、phase、事件类型、调用方、持续时间、退出状态、
artifact，以及 provider 返回的原始 usage。聚合器不得覆盖或删掉失败尝试。

一次 `evaluate`、`verify` 或 `trace` 请求记为一个 logical oracle query；`inspect`、`run`
和对已有 trace 的确定性 `localize` 不计。执行前即可判定必然失败的调用（工具在本条件不可用、
参数不合法、`evaluate` 没有可评估的 run）不计；已到达验证服务后被拒绝的调用（如产物与当前
candidate 不符）仍计数。重复查询同一 candidate 仍计数，同时另记
唯一 candidate query 数量。trial 同时受 oracle query 和 wall time 限制，先到者停止。

Logical Query Cost 与 Physical Evaluation Cost 分开保存，不能互相替代：

- Logical Query Cost：`evaluate / verify / trace` 的调用总数、分工具计数和唯一 candidate 数。
- Physical Evaluation Cost：实际启动的 source/target 进程数、这些进程的墙钟秒数，以及
  device 为 CUDA 的进程墙钟秒数 `cuda_execution_seconds`。
- `cuda_execution_seconds` 是 CUDA 配置进程占用的墙钟上界，不声称等于 GPU active time。
- `evaluate` 只读已有产物，因此增加 logical query，但不重复累计产生该产物的执行成本。
- `run` 可能实际启动 source/target，因此累计 physical cost，但不增加 logical query。
- Agent 阶段与隐藏 official evaluator 的 physical cost 分栏保存，同时提供总计。

聚合结果还从原始事件确定性导出 `tool_calls_by_name`、`tool_time_seconds_by_name`、
`physical_evaluation_cost_by_tool`、candidate revision 数，以及首次成功 `run`、首次开发
`verify` 通过和 official verify 完成时间。按工具 physical cost 是累计执行成本的事件间
差值，不改变 Logical/Physical 两类成本的定义。

## 正确性与停止

Agent 可以用自己的代码工具和普通测试修改、运行 candidate，但 TorchPort 证据只能通过 Harness
的六个工具取得。正式成功只由 trial 结束后的独立 `torchport verify` 决定。结束时 Harness 先
冻结完整仓库、文件 manifest 和 Git diff，再把隐藏 task 的 target 指向该快照。最终 evaluator
使用独立运行目录，Agent 不得修改其 fixture、容差或 oracle。

正式 candidate 是完整仓库，不是单独的 `submission.py`。当前 held-out 工作区使用薄 `run.py`
作为统一入口；模型实现、配置、依赖和辅助模块仍保留在正常仓库结构中。任务专用隐藏测试应覆盖
正常导入、配置、checkpoint 加载和推理路径。通用 Harness 只负责冻结与审计，不能仅靠文件数量
证明迁移质量。

停止原因使用固定枚举：

- `agent_completed`
- `wall_budget_exhausted`
- `oracle_query_budget_exhausted`
- `token_budget_exhausted`
- `turn_budget_exhausted`
- `agent_stopped`
- `infrastructure_error`

`stop_reason` 只描述 Agent 循环为什么结束。`result.json.passed` 只取独立 evaluator
的结果；即使预算耗尽，结束时留下的 candidate 通过正式验收仍记为成功。

基础设施错误不自动算方法失败，也不能静默删除。协议预先规定最多重跑一次；每次 attempt
都保留并标记有效性，重跑使用新的 attempt id 和一份从冻结 seed 复制的干净 candidate。
所有 attempt 仍属于同一个预注册 trial，因此不扩大 Success@Budget 分母。有效的模型失败、
预算耗尽和正式验收失败不重跑。

## 核心指标

### Success@Budget

```text
正式验收通过的 trial 数 / 预先排定且有效的 trial 总数
```

分母在实验开始前冻结。失败、预算耗尽和 Agent 主动停止都保留在分母中。结果同时
报告 `successes / total`，不只给百分比，也不按 task、round 或最好结果去重。

### Time-to-Success

从 trial 启动到首次独立 evaluator 通过。一个 trial 内后续成功不能替换首次成功
时间。失败 trial 不记作零，也不塞入成功样本中位数；报告成功样本中位数时必须同时
给出 `successes / total` 和失败的停止原因分布。

### Query-to-Success

固定 query 上限为 `K` 的 trial 报告正式 `Success@K`；比较多个 K 时使用分别预注册的
trial，不能从一次 K=10 的最终结果倒推 Success@1。另报告开发 verifier 首次通过所需的
query 数，但它不能替代隐藏 evaluator 的正式成功。失败 trial 不填成 `K`，而是保留为
未成功，并与停止原因一起报告。wall time 仍是硬限制，避免 Agent 无限编辑而不查询。

### Token 与费用

保留 provider 原始分类：input、cached input、output、reasoning 及 provider total。
派生总量不得覆盖原始值。只有存在 provider 账单或带日期、模型和币种的价格快照时
才计算费用；否则 `cost` 为 `null`。

### 时间与 compute

分别记录 Agent wall time、TorchPort tool time、最终 evaluator time。进程级 CUDA 墙钟
按上面的 `cuda_execution_seconds` 报告；只有能获得可靠
设备利用率或进程级测量时才报告 GPU compute；不能把 trial 墙钟时间改名为 GPU
minutes。

## 最小性能协议

性能只对独立验收通过的 candidate 运行，使用 `mode=normal`，关闭 trace。性能结果
默认不影响正确性结论。

每个 task 固定 batch、warmup 次数、测量次数、同步点和缓存策略，并记录：

- `cold_start_seconds`：空 trial 缓存下，进程启动到第一次同步输出完成。
- `warm_start_seconds`：复用同一 trial 编译缓存的新进程首次同步输出。
- `steady_latency_ms`：模型已加载并完成 warmup 后的同步 latency，报告 median 和 p90。
- `throughput_per_second`：固定 batch 除以 median latency。
- `memory`：没有可靠跨框架测量时明确为 `unsupported`。

PyTorch 使用 CUDA synchronize；Jittor 使用等价的全局同步。计时边界必须包围同步后的
真实完成点。两端在同一空闲 GPU 上串行测量，不并行争用资源。

## 实施顺序

1. 冻结 Gateway、反馈裁剪、query 计数与隐藏认证边界，先用 tiny 做协议测试。
2. 为成熟代码 Agent 编写薄外部适配层；它只映射 Gateway 发布的六个 schema，并把模型、turn
   和 usage 原始事实写入事件流，不接管 Agent 的普通代码工具。
3. 编写只读聚合器，保留所有 trial，输出 Success@Budget、Time-to-Success、Token
   和停止原因。
4. benchmark 冻结后，为真实任务建立开发与隐藏 TaskSpec，并完成小规模 pilot。
5. pilot 冻结后再增加 performance.json；内存测量留到有可靠实现时。

实验 controller 与聚合器放在 `experiments/harness/`，具体 manifest 和产物放在
`experiments/p2/`。controller 在首次请求 Agent 前验证完整 trial 矩阵与冻结 artifact，
生成 `run-plan.json`、`runtime-snapshot.json` 和可恢复的 `run-state.json`。TorchPort 核心
只在现有 TaskSpec/manifest 中补充确实需要的运行事实，不增加第二套执行器、数据库或
统计框架。

正式实验还必须由外部隔离 worker 生成 attestation。controller 只校验并冻结该证据；
它不把 JSON 声明当作隔离实现。attestation 至少证明宿主文件系统与凭据隔离、candidate
是唯一可写挂载、source 与冻结输入只读、隐藏验收材料未挂载给 Agent，并记录 runtime 与
不可变 image digest。缺失任一条件时不得生成正式 run-plan。
正式实验的 attestation 还必须证明：Agent 看不到 Harness/TorchPort 控制代码、人工迁移
校准实现和其他 candidate 工作区；每个条件从新会话启动，不挂载先前 trial 的历史或持久记忆，
也无法通过会话工具访问其他 trial。
公开网络、网页搜索和远程代码搜索保持可用，两组使用相同的普通工具配置，并冻结工具清单摘要。
公开同题实现可以被 Agent 检索和引用；记录 URL、版本或提交号，结论解释为通用 Agent 在公开
检索条件下的迁移效果。外部适配层保存网页和代码搜索事件；这些普通工具调用不计入 TorchPort
Logical Query Cost。公开参考的使用情况单独汇报，不使 trial 无效。
这些字段由 worker 对实际可见挂载和会话配置生成，不能由实验作者手填 `true` 代替隔离。
prompt 中的访问规则用于指导 Agent；正式隔离以 worker 实际执行的边界为准。

## 首轮环境验收

正式 benchmark 暂不固定。先只用 tiny 验收三种条件，确认路径无法越界、隐藏认证不可
调用、字段裁剪没有泄漏、query 无法超支、所有工具响应均有哈希产物。tiny 只验证实验
管线，不能代表真实迁移收益。通过后再冻结任务集合、K、wall time 和重复次数。

## 最小闭环结果（2026-10-02）

`tiny-smoke` 已在同一块 RTX 3080 Ti 上串行完成 `verdict`、`diagnostic`、`localized`
各一次预注册 trial。三组使用相同初始 candidate、prompt、模型和预算，均通过独立
official verify；每组各调用一次 `evaluate` 和一次 `verify`。聚合结果为 3/3 valid、
3/3 success，总计 55,883 provider total tokens、39 次 CUDA 配置进程执行和
1,146.75 CUDA 进程墙钟秒。

通过后的 candidate 使用 `mode=normal`、关闭 trace 完成性能协议：
`warmup=10`、`iterations=30`，输出 cold/warm start、median/p90 latency 和 throughput；
memory 明确记录为 `unsupported`。冻结配置、便携汇总和性能报告位于
`experiments/p2/tiny-smoke/`，原始 trial 目录保留在本地但不进入 Git。

该结果只证明旧版 P2 管线、成本边界、隐藏验收和性能产物闭环曾经可运行。其直接模型 API
adapter 已经移除，不能作为当前成熟 Agent Harness 的回归结果。
tiny 初始 candidate 本来就正确，且每组只有一个 replicate，因此不能据此声称某种反馈
粒度提高迁移成功率；正式研究结论仍需冻结真实 benchmark 后获得。
