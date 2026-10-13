# 架构说明

Jittor Harness 是对外项目名；`torchport` 是其内部执行与验证核心的 Python 包和 CLI 名。
两个名字分别表示完整 Agent 工作流和稳定的底层接口，不存在两套执行器。

## 依赖方向

```text
外部代码 Agent
        │ 编写或修改 candidate，生成显式 task
        ▼
TorchPort CLI
        │
        ├── contracts：冻结 TaskSpec、双端内容身份与 oracle key
        ├── runtime：隔离子进程、环境、超时、日志、缓存与物理成本
        ├── capture：采集输入、状态、最终输出和可选 trace
        ├── comparison：独立比较张量、包络、模块配置和任务指标
        ├── verification：运行 candidate、管理 oracle 并给出唯一判定
        └── diagnosis：公开检查、trace 与首次分歧定位
```

TorchPort 不读取未知仓库后自动改写代码。仓库理解、环境检查、迁移策略和代码修复属于 Agent；
TorchPort 只处理显式任务、可复验证据和经过准入的迁移问题条目。Agent 通过 CLI 或 Harness
使用 TorchPort，核心不绑定某个 Agent 产品，也不维护第二套工具协议。

核心依赖保持单向：`contracts → runtime/capture → comparison → verification`；
`diagnosis` 复用同一运行、采集、比较和 oracle 能力，只增加诊断编排。内部和外部调用都
使用这些 canonical package，不再保留平面的历史导入入口。

## 稳定契约

### TaskSpec

TaskSpec 明确 source 和 target 的仓库、入口、模型、构造参数、Python、设备和环境，
并冻结输入、checkpoint、随机种子、容差、重复次数和超时。代码、配置、权重或协议
改变都会改变内容身份。

### RunManifest

每次运行创建独立目录，记录 task、trial、side、代码身份、执行参数、开始和结束时间、
exit code、timeout、实际缓存目录及产物。只有本次子进程成功并写出 capture，manifest
才标记为可用；旧目录中的 NPZ 不能补偿本次失败。

### EvaluationResult

验收按 runnable、structural、numerical 三层判定。输入哈希、目标参数完整性、输出
结构和 FP32 数值比较都必须通过；证据缺失不能默认成功。Strict 结果用于诊断，默认
不覆盖 FP32 推理协议的最终结论。

## 验证工作流

```text
task.yaml
    │
    ├── source 重复执行 ──> 稳定 oracle
    │
    └── target 重复执行 ──> candidate manifests
                               │
                               ▼
               输入/参数/结构/FP32/Strict 比较
                               │
                               ▼
                         verify result
                               │
                         Agent / CI / 报告
```

普通验收使用 `mode=normal`，不会回放 source 的中间算子输出。`mode=trace` 属于诊断
运行，必须单独标记，不能签发普通验收成功。系统没有算子输出干预运行模式。

## CLI

| 命令 | 职责 |
| --- | --- |
| `doctor` | 检查本机框架与设备环境 |
| `inspect` | 查看公开 TaskSpec，不读取或创建 oracle |
| `run` | 按 task 执行 source 或 target，输出 RunManifest |
| `evaluate` | 只读既有 oracle 和 candidate 产物并判定 |
| `trace` | 以相同输入和状态采集双端中间证据 |
| `localize` | 从已有 trace 报告提取首次分歧，不重新执行模型 |
| `verify` | 冻结 oracle，重复运行 candidate 并生成唯一结果 |
| `compare` | 独立比较两份 NPZ 张量证据 |

`trace` 和 `localize` 组合同一执行器、采集器和比较器，不建立第二套 Harness。P2 的反馈裁剪、
预算、事件和成本计数位于 `experiments/harness/`，不进入判断核心。

成熟代码 Agent 直接使用自己的文件、Shell、Python、Git、测试和网络能力。Harness 只提供六个
TorchPort 工具：`tooling.py` 定义工具 schema，`feedback.py` 实施反馈裁剪，`gateway.py` 记录
调用并守住隐藏认证边界，`candidate.py` 冻结完整候选仓库、manifest 与 Git diff，
`orchestration/` 冻结并执行 run plan，`reporting/` 只读聚合结果。最终 `verify` 针对冻结快照
运行，不针对 Agent 仍可继续修改的工作目录运行。

## 有证据的迁移问题库

问题库是验证结果的下游消费者，不参与 `passed` 判定。条目按 API 语义、状态加载、输入
流水线、执行模型、数值后端和环境等可观察层次分类，不判断失败应归因于框架还是 Agent。
每条记录保存触发条件、差异、最小复现、修复、before/after 证据、适用范围和未知项。
首版使用普通 JSON 和确定性字段过滤；不会自动执行修复，也不会把首次分歧升级为根因。
完整准入规则见 [P3 有证据的迁移问题库](p3-verified-issues.md)。
