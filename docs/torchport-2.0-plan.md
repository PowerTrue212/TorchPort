# TorchPort 2.0 路线图

日期：2026-10-03。P0、P1 和 P2 最小闭环已经完成。后续主线调整为：

> **可信验证工具 + 有证据的迁移问题库**

P4 反事实干预已经删除，不设替代阶段。现有 `trace` / `localize` 保留为按需辅助能力；
不再把复杂定位或自动根因诊断作为必须完成的独立阶段。原 P5 提前为新的 P3。

## 项目边界

Agent 负责阅读仓库、理解代码、提出假设并修改 candidate。TorchPort 提供 Agent 无法
凭空生成的两类事实：

1. 在冻结任务和协议下获得的可复现执行、数值比较与验收证据。
2. 来自真实迁移、保留适用边界和未知项、且能复验的迁移问题经验。

第一版限定 PyTorch → Jittor、单机 CPU/CUDA 和推理。TorchPort 不负责未知仓库的
入口猜测、自动代码转换、通用图 IR、自动根因裁决或自动执行知识库中的修复命令。

研究问题调整为：

> **提供经过验证的迁移经验，能否让 Agent 在未参与条目整理的新模型上少犯同类错误、
> 更快完成迁移？**

## 最小架构

```text
Agent / DSH / CLI
       │ 修改 candidate，提交显式 task
       ▼
task + execution ──► capture ──► oracle / candidate artifacts
                                             │
                                             ▼
                                parity + verification
                                             │
                              result + optional trace
                                             │
                                  verified issue entries
```

| 模块 | 唯一职责 |
| --- | --- |
| `task.py` | 加载、校验和冻结任务、协议及双端内容身份 |
| `execution.py` | 子进程、环境、超时、日志和 trial 缓存隔离 |
| `capture.py` | 双框架加载、状态核验、最终输出与显式 trace 采集 |
| `oracle.py` | reference oracle 生命周期、candidate 执行和 verify 编排 |
| `parity.py` | 与执行器无关的数组比较 |
| `verify.py` | runnable → structural → numerical 的唯一验收判定 |
| `diagnosis.py` | 公开 task 检查、双端 trace 和首次分歧提取 |

问题库先使用普通 JSON、fixture 和已有运行产物。只有出现第二个真实调用方时才增加
运行时检索模块；不先建数据库、向量索引、插件框架或新的执行器。

## 三个稳定契约

### TaskSpec

冻结 source/target 仓库、入口、构造参数、Python、device、输入、checkpoint、随机种子、
容差、重复次数和 timeout。代码、配置、权重、环境或协议改变都会改变内容身份。

### RunManifest

每次运行记录 task/trial/side、代码身份、执行参数、时间、exit code、timeout、实际缓存
和产物。运行模式只有 `normal | trace`。只有本次进程成功写出 capture 才能使用产物；
旧目录中的文件不能补偿本次失败。

### EvaluationResult

`runnable`、`structural`、`numerical` 分层报告，唯一 `passed` 由冻结策略推导。输入摘要、
参数映射和加载后值、输出结构及 FP32 数值证据缺一不可。Strict 中间轨迹属于诊断，
默认不覆盖最终 FP32 结论。只有 `mode=normal` 可以签发普通验收成功。

## CLI 分工

| 命令 | 语义 |
| --- | --- |
| `inspect` | 查看公开 TaskSpec，不读取 oracle |
| `run` | 按 task 执行一端并写 RunManifest，不判断迁移成功 |
| `evaluate` | 只读已有 oracle/candidate 产物并比较，不启动模型 |
| `verify` | 冻结 oracle，重复执行 candidate，给出唯一验收结果 |
| `trace` | 对同一任务重新采集中间证据，不修改普通验收结果 |
| `localize` | 从已有 trace 提取首次分歧，不重新执行模型 |

这些命令组合相同的 task、execution、capture、parity 和 verify 函数。Agent 接入优先使用
CLI；核心不新增 SDK、MCP 服务或第二套 passed 判定。

## P0：验证可信性（完成）

- 普通执行和 trace 都会真实调用 target，不回放 source 中间输出。
- 错误算子、输入摘要缺失、参数缺失/多出/加载后值错误和输出结构错误均有负例。
- oracle 以 source 内容与协议身份失效重建；不同 target 不共享 candidate 产物。
- trace 运行不可签发普通验收成功。

## P1：独立 Harness 与 Evaluation（完成）

- 显式 task → 稳定 oracle → candidate 运行 → 唯一 result 已闭环。
- 双端可以使用独立 Python 环境；执行器统一 cwd、device、timeout、日志和缓存隔离。
- 正式 candidate 按预先指定次数重复执行，全部通过才成功，不取最好一次。
- DSH 插件消费 TorchPort 结果，不重新推导另一套成功条件。

## P2：受控 Agent 评测与成本（最小闭环完成）

完整字段与 pilot 结果见 [P2 评测协议](p2-evaluation-protocol.md)。关键边界已经冻结：

- `verdict`、`diagnostic`、`localized` 只改变允许返回的字段与工具能力。
- Diagnostic 只提供阶段、失败码、类别和聚合数值，不包含 module/file/line/operator/
  first divergence；这些只属于 Localized。
- Logical Query Cost 单独统计 evaluate/verify/trace 调用。
- Physical Evaluation Cost 单独统计 executions、wall time 和 CUDA 配置进程秒数。
- Gateway 提供受控 `probe(path, args=[])`；它只运行 target 内 Python 脚本，计 physical
  cost，不计 logical query，也不能签发成功。
- 聚合器按工具输出调用次数、工具时间、physical cost、candidate revision 与关键成功
  milestone。
- 性能通过独立的 `mode=normal` 运行测量，默认不改变正确性结论。

CLIP/RAFT pilot 的任务物化、public/hidden TaskSpec 和六个预注册 trial 已冻结；正式多
replicate 结论仍需在具备宿主文件与网络隔离的 worker 上运行。tiny GPU smoke 只证明
评测管线可运行，不能说明某种反馈粒度更有效。

2026-10-03 的 RTX 3080 Ti 基础设施验收进一步确认：两个 task 的 CUDA probe、trial 内
Jittor 缓存、public/hidden oracle 与物理成本记录均可运行；CLIP 历史候选 public/hidden
各 3/3 通过，RAFT 历史候选在两组输入上均得到真实 `PARITY_FAILED`。该结果只验证
harness 能区分运行成功与数值成功，不是 verdict/diagnostic/localized 的 Agent 对照结果。

## 基础定位：保留为辅助能力

`trace` / `localize` 当前只提供真实中间证据和首次分歧线索，不宣称根因。后续只有真实
问题暴露明确缺口时才补 scope、调用序号或采集预算。不会建设自动根因引擎、传播图、
反事实干预或自动修复器。

这条能力的验收边界保持不变：探针可能影响懒执行、融合和同步；不同覆盖范围、门限或
张量定义的 trace 不能直接比较；定位结果本身不能证明修复有效。

## P3：有证据的迁移问题库（下一阶段）

问题库不判断失败属于“框架固有问题”还是“Agent 引入错误”。这类责任归因往往由框架
语义、代码假设和具体修改共同决定，边界不稳定，也不能帮助复现。条目只按可观察的问题
层次分类，例如 API 语义、状态加载、输入流水线、执行模型、数值后端和运行环境。

每个条目必须描述触发条件、可观察差异、最小复现、修复及复验、适用边界和仍未知的
部分，并链接同协议的 before/after 证据。Agent 在某次 trial 中采取了什么错误修改仍由
实验事件记录保存，不进入兼容性事实的主分类。一次成功只能标为案例级验证；只有跨独立
案例复现后才能提升为 generalized。旧 `compatibility/rules.json` 继续作为历史规则，未
补齐证据前不自动升级。

详细 schema、准入规则、held-out 划分和最小实现批次见
[P3 有证据的迁移问题库](p3-verified-issues.md)。

P3 的评测使用未参与条目整理的新任务。无问题库和有问题库两组使用相同模型、prompt、
初始 candidate、预算、工具和 evaluator；按仓库、修订和问题族隔离，避免把记住原修复
误当作迁移能力。主要指标包括 Success@Budget、同类错误复发率、Time-to-Success、
Logical Query Cost、Physical Evaluation Cost、token 成本及负迁移率。

## 控制代码体积

1. 保持唯一子进程执行路径、数组比较内核和验收判定。
2. 新命令只能组合已有函数；没有第二个真实实现需求时不建抽象工厂或插件注册器。
3. 普通 JSON/NPZ/JSONL 足够；不引入 schema 框架、数据库、embedding 或在线训练。
4. 模型特例放 task runner，实验统计放 `experiments/`，迁移决策留给 Agent。
5. 问题库检索先做确定性字段过滤；存储的命令和 patch 永不自动执行。
6. 每个阶段评审新增模块、依赖和重复逻辑，优先删除没有真实调用方的抽象。

## 接下来的实施批次

1. 冻结 P3 条目 schema、证据准入、状态提升和 held-out 防泄漏规则。
2. 从真实迁移中选择两个高价值、可复现的问题，分别保存最小失败 fixture、修复及普通
   verify 的 before/after 证据；不要求对失败责任做二分归因。
3. 条目达到案例级验证后，再加入简单、只读、确定性的字段检索。
4. 冻结未见任务，运行无问题库/有问题库对照 pilot，检查收益和负迁移。
5. 只有真实案例显示基础定位不足时，才扩充 trace/localize；不恢复已删除的 P4。

P0–P2 构成可信执行、验收和效果评测底座。P3 让一次迁移产生的可靠经验能够被下一次
迁移复用，同时保留未知项，避免把相关性、一次成功或 Agent 猜测包装成通用根因。
