# 项目目录说明

TorchPort 是迁移 Agent 的执行与验收工具。核心代码只处理显式 TaskSpec、隔离执行、
证据采集和唯一判定；仓库理解与代码修改由 DSH 或其他 Agent 完成。

## 核心代码

| 路径 | 职责 |
| --- | --- |
| `src/torchport/task.py` | 加载、规范化并冻结 TaskSpec 和双端内容身份 |
| `src/torchport/execution.py` | 子进程、环境、超时、日志和 trial 缓存隔离 |
| `src/torchport/capture.py` | 双框架模型加载、状态核验、输出与可选 trace 采集 |
| `src/torchport/oracle.py` | oracle 生命周期、candidate 执行和 verify 编排 |
| `src/torchport/parity.py` | 与执行器无关的张量比较 |
| `src/torchport/verify.py` | runnable、structural、numerical 的唯一判定 |
| `src/torchport/diagnosis.py` | 公开 task 检查、双端 trace 和首次分歧定位 |
| `src/torchport/cli.py` | `inspect/run/evaluate/trace/localize/verify/compare` 的薄入口 |

不在核心中加入未知仓库扫描、入口猜测、自动代码改写、模型专用适配或 Agent 决策。

## 仓库目录

| 路径 | 内容 |
| --- | --- |
| `tests/` | 核心契约、进程生命周期、采集、比较和负例测试 |
| `benchmarks/tasks/` | 公开、显式、可复现的 TaskSpec 与小型双端 fixture |
| `integrations/dsh/` | DSH 工具注册、轨迹与报告适配，不包含迁移器 |
| `compatibility/` | P3 的历史兼容性规则、迁移问题条目和已验证修复证据 |
| `experiments/` | 汇总数据、绘图代码、报告和可复现实验记录 |
| `experiments/harness/` | P2 受控 Agent 工具、反馈策略、预算和事件记录 |
| `CLIP/`、`RAFT/`、`apt/` | 独立迁移实验协议及本地工作区 |
| `submissions/` | 需要长期保存的 Agent candidate 快照 |
| `deliverables/` | 面向交付的报告与演示文稿 |
| `scripts/` | 安装、复现和远程开发工具 |
| `docs/` | 架构、路线图和开发说明 |

## 本地产物

`.venv/`、`.tmp/`、`.torchport/`、`.codex-build/`、`.codex-finalizer/`、缓存、权重和
实验工作区都是可重建或机器相关的内容，由 `.gitignore` 排除。需要长期保留的实验
结论应提炼到 `experiments/results/`、`submissions/` 或 `deliverables/`。

## 控制代码体积

新增命令只能组合现有 task、execution、capture、oracle、parity 和 verify 函数。
性能与成本信息扩展现有 manifest/result，不新建另一套运行器。模型特例进入 task
runner；实验统计进入 `experiments/`；迁移策略留给 Agent。
