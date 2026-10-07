# TorchPort 跨对话工作说明

更新时间：2026-10-07  
当前开发分支：`torchport-2.0-core`

这份文件是新对话和新 Agent 的项目入口。它只保存当前方向、不可破坏的边界和接手顺序。
详细设计、实验协议和原始证据以链接文件为准。阶段目标改变时更新本文件，不在这里堆积逐次
运行日志。

## 接手时先做什么

1. 运行 `git status --short --branch` 和 `git log --oneline -5`，确认分支、已有修改和最近提交。
2. 阅读本文件，再按任务阅读下方链接。不要从旧 prompt 或单份历史报告反推当前设计。
3. 修改前确认哪些文件已经由用户或其他 Agent 改动；不要覆盖无关的未提交内容。
4. 先复现当前结论，再修改实现。正式实验结果必须保留原始 JSON/JSONL/NPZ 证据。
5. 完成后运行与改动对应的测试，并说明正式结论、诊断结论和仍未知的部分。

## 项目要做什么

TorchPort 服务于 PyTorch 到 Jittor 的仓库迁移。迁移 Agent 负责理解仓库、修改代码和提出
修复；TorchPort 负责提供 Agent 不能凭空生成的事实：

- 在冻结输入、权重、环境和协议下真实运行 PyTorch 与 Jittor。
- 检查输入、参数、输出结构和数值结果，给出唯一且可复验的验收结果。
- 在验收失败时提供分级证据，并记录获得这些证据的实际成本。
- 把真实迁移中反复验证过的问题整理成可复现的问题条目，供后续任务参考。

项目定位和对外说明见 [README](README.md)。核心结构见
[架构说明](docs/architecture.md)，阶段安排见 [TorchPort 2.0 路线图](docs/torchport-2.0-plan.md)。

TorchPort 当前不是自动代码转换器，也不负责替 Agent 猜入口、判断最终根因或自动执行修复。
公开 CLI 没有 `migrate` 命令。仓库中的历史迁移代码或旧 prompt 不是当前核心工作流，不要在
没有明确需求时重新接入或扩张。

## 当前有两条 Agent 接入路线

### 成熟的代码 Agent

DSH、Codex 等成熟 Agent 应保留正常的仓库浏览、搜索、编辑、Shell、Python、测试和网络
检索能力。TorchPort 作为附加验证工具接入，环境负责隔离 hidden task、oracle、凭据和最终
evaluator，并记录调用与成本。

DSH 插件只映射六个命令：`inspect`、`run`、`evaluate`、`verify`、`trace`、`localize`。
安装和工作方式见 [DSH 插件说明](integrations/dsh/README.md)。用户自己运行 `dsh web`；项目
Agent 不代替用户启动它。远端 GPU 连接只从 `~/.tp_ssh.env` 读取，密码和主机信息不能写入
仓库。

### 直接调用模型 API 或本地代码模型

这类模型没有现成的代码操作环境，需要 `experiments/harness/` 提供文件、搜索、编辑、执行、
测试和 TorchPort 工具，并统一记录预算和事件。现有 Harness 已能完成最小闭环，但狭窄工具
接口不能作为成熟 Agent 的公平替代。正式比较前必须确认模型拥有完成真实迁移所需的代码操作
能力。

两条路线可以分别实现，但实验中的唯一自变量应是 TorchPort 反馈粒度。不能让一组使用完整
代码工具，另一组只能使用少数受限文件工具，然后把差异归因于 TorchPort。

## 已冻结的实验边界

P2 比较三种反馈：

| 条件 | Agent 可见内容 |
| --- | --- |
| `verdict` | 是否通过 |
| `diagnostic` | 阶段、通用失败码、类别、聚合数值、结构和权重状态 |
| `localized` | 在 diagnostic 基础上增加 trace 和首次分歧证据 |

`diagnostic` 不得出现 module、file、line、operator、张量名、first divergence 或暗含位置的
自由文本。`localized` 才能返回 scope、invocation、operator、first divergence、输入输出证据
和 error jump。详细字段见 [P2 评测协议](docs/p2-evaluation-protocol.md)。

成本必须分开记录：

- Logical Query Cost：`evaluate`、`verify`、`trace` 的调用次数、分工具计数和唯一 candidate 数。
- Physical Evaluation Cost：实际执行次数、进程墙钟时间和 CUDA 配置进程墙钟时间。

两者不能互相替代。文件读取、代码修改和普通测试可以另行计数，但不能混入 oracle query。

## 已经完成什么

- P0：TaskSpec、source oracle、candidate 身份、负例和唯一验收结果已经闭环。
- P1：双端独立环境、运行记录、重复验收、DSH 六命令插件已经闭环。
- P2：反馈裁剪、DeepSeek API adapter、事件记录、成本聚合和 tiny GPU smoke 已完成最小闭环。
- CLIP/RAFT：已经完成基础设施与接口覆盖审计；历史结果不能直接用于正式效率排名。
- CamoFormer：已完成一次真实 COD 仓库迁移，并暴露出 TF32、固定容差和 target 数值波动对
  验收协议的影响。
- 模块隔离：`trace` 可以记录 source 真实局部输入，直接运行同名 target 模块，并分别报告
  固定容差、source 噪声标定和 target 重复性证据。

核心接口覆盖审计见 [P1/P2 Agent 接口审计](docs/audits/p1-p2-agent-interface-audit.md)。
目录职责见 [项目目录说明](docs/project-layout.md)。

## 现在正在做什么

当前主任务是把 P2 从“管线能运行”推进到“可以公平评测真实迁移效果”：

1. 校准 CamoFormer 暴露出的数值验收问题，区分模型实现错误、后端正常数值差异和重复运行
   波动；在证据充分前保留正式失败。
2. 为成熟 Agent 准备不削弱其代码能力的实验环境，使 verdict、diagnostic、localized 三组只在
   TorchPort 反馈上不同。
3. 在相同初始 candidate、模型版本、提示词、硬件和预算下运行多次真实任务，比较成功率、
   用时、Token、Logical Query Cost 和 Physical Evaluation Cost。
4. P2 协议稳定后，再推进 P3 的可复现迁移问题库。P3 不按“框架固有问题/Agent 引入错误”
   二分，只记录可观察问题、触发条件、复现、修复、适用边界和未知项。

P3 的条目格式和准入规则见 [有证据的迁移问题库](docs/p3-verified-issues.md)。复杂自动根因诊断
和原 P4 反事实干预不在当前路线中。

## CamoFormer 当前结论

CamoFormer 候选已经可以完整运行，参数与结构检查通过，但冻结协议的正式数值验收失败。
关闭 source TF32 后失败元素大幅下降；这证明冻结 source 数值路径是主要影响因素之一，不能
证明所有剩余误差都无害。

模块隔离实验覆盖 260 次模块调用、271 个输出张量（source 关闭 TF32 的诊断协议）：

- 噪声标定规则：`err(target, fp64) <= k * max(err(source, fp64), err(source, alt)) + 1e-7`，
  按 max_abs 与 p99 分别判定，所有模块（含组合模块）参与判定，默认 `k=8`。
- 正确候选：随机权重最大 ratio 1.84，PVTv2-B4 真实 encoder 权重（decoder 随机，
  `complete=false`）最大 ratio 3.89，均通过。
- 负对照：`ln_eps`、`gelu_tanh`、`mask_residual` 均失败（最小 ratio 约 27），首个失败模块与
  注入位置一致；只看叶子模块会漏掉 `mask_residual`。`l2_eps` 在当前输入上无数值效果。
- source（PyTorch 默认设置）同输入重复逐位一致；target（Jittor 1.3.9.14 / CUDA 12.8 / sm_86）
  不确定：整网同进程重复约 98% 输出元素逐位不同，max_abs 约 3e-3~5e-3。`no_fuse`、
  `enable_tuner=0`、`jit_search_kernel=0`、`use_tensorcore=0` 均无效；`conv_opt=0` 更差。
  逐模块双跑时首个不确定模块是 `encoder.block1.0.attn.sr`（Conv）。kernel 名确认：
  `conv_opt=1` 走 Jittor 自身 reindex + 非末维归约卷积（单卷积运行间 6e-7~1.4e-6）；
  `conv_opt=0` 走 `cudnn_conv`（单卷积逐位确定，但整网反而更差，原因未查清）。是否为
  原子加尚待读源码确认。
- `k=8` 只用 2 种权重设置和 4 个负对照估计，仍是探索参数，不能替代正式验收。
- V100（sm_70，无 TF32，PyTorch 2.5.1 / Jittor 1.3.9.14 + CUDA 11.3）复现：随机权重
  最大 ratio 3.17 通过；`ln_eps`/`mask_residual`/`gelu_tanh` 失败（最小 ratio 25.97），`l2_eps`
  仍无数值效果；source 逐位确定、target 不确定。随机权重整网 max_abs 达 1.03，见下条解释。
- 任务指标（V100，官方完整 checkpoint，COD10K/CAMO/NC4K/CHAMELEON 共 6473 张，官方
  `evaltools`）：PyTorch 复现论文；Jittor 1100/1100 参数加载，四数据集 S/wF/E/MAE 与
  PyTorch 差值均约 1e-7，逐图最大差 1/255，阈值 128 翻转像素 < 1e-6。随机权重整网大误差
  来自未训练网络的条件数放大，不是实现错误。Jittor 推理耗时约为 PyTorch 的 4.2 倍。
  证据：`CamoFormer/outputs/task-metrics/`。

完整说明和原始证据入口见 [CamoFormer 实验记录](CamoFormer/outputs/dsh-web-camoformer/README.md)
和 `CamoFormer/outputs/dsh-web-camoformer/probe/` 下的 trace、negctl 与 `probe_jt_*` 证据。

仍未完成：Jittor 不确定性根因（源码级确认）、`conv_opt=0` 整网更差的原因、Jittor 推理
性能差距、noise-calibrated 是否升级为正式协议。若建立新容差或关闭 TF32 的新协议，必须作为新协议
重新评测，不能改写原冻结协议的失败结果。

## 不能破坏的实现规则

1. `verify` 是普通验收的唯一最终判定入口；`trace`、`localize` 和噪声标定只提供诊断证据。
2. 证据缺失、运行失败、参数不完整或输出结构不同不能默认成功。
3. source 代码、配置、输入、权重、环境或协议改变后，oracle 必须失效重建。
4. 不同 target 不能共享 candidate 运行产物；candidate 多参数、少参数或值加载错误都必须失败。
5. hidden task、oracle 和 evaluator secret 不能进入 Agent 可见工作区。
6. 不把随机初始化前向对齐写成真实数据集任务效果；任务级成功需要真实 checkpoint 和数据集指标。
7. 不把首次越过容差的位置写成已经确认的根因。
8. 新功能优先组合现有 `task/execution/capture/oracle/parity/verify/diagnosis`，不要建立第二套执行器、
   比较器或成功判定。

## 关键目录

| 路径 | 用途 |
| --- | --- |
| `src/torchport/` | 冻结任务、双端执行、证据采集、比较和验收核心 |
| `tests/` | 核心契约与负例测试 |
| `integrations/dsh/` | 成熟 DSH Agent 使用的六命令插件 |
| `experiments/harness/` | 直接调用模型 API 时使用的实验控制器 |
| `benchmarks/tasks/` | 小型、显式、可复现的任务定义 |
| `CLIP/`、`RAFT/`、`CamoFormer/`、`apt/` | 真实迁移工作区、提示词和证据 |
| `compatibility/` | 历史规则和后续 P3 问题条目 |
| `experiments/results/`、`submissions/`、`deliverables/` | 需要长期保存的结果和报告 |

大型工作区、缓存和临时产物不应直接并入核心代码提交。需要保留的实验结论应提炼为报告和
可追溯证据路径。

## 常用检查

```powershell
# 核心 Python 测试
.\.venv\Scripts\python.exe -m pytest -q

# 检查本次改动的 Python 文件；全仓旧绘图脚本可能仍有独立 Ruff 问题
.\.venv\Scripts\ruff.exe check <changed-files>

# DSH 插件
npm test --prefix integrations/dsh

# 查看 CLI
.\.venv\Scripts\python.exe -m torchport --help
```

远端 GPU 开发说明见 [远程 GPU 文档](docs/development/remote-gpu.md)。调用远端前先确认
`~/.tp_ssh.env` 存在，禁止打印或提交其中的密码。

## 当前优先级

1. 完成 CamoFormer 多输入、真实 checkpoint 和重复性校准报告。
2. 把成熟 Agent 的正常代码能力与 TorchPort 反馈控制明确分层，并修订与该决定冲突的旧 P2
   文档。
3. 冻结首轮真实 benchmark、replicate、预算和停止条件。
4. 运行 verdict / diagnostic / localized 对照；历史 CLIP/RAFT/APT 结果只作案例证据，不混入
   新实验的效率统计。
5. 根据真实失败决定是否补充 trace；没有案例需求时不增加抽象、数据库或自动根因系统。
