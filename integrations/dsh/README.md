# TorchPort DSH 插件

该包把 TorchPort 的确定性分析、迁移和数值验证能力注册为 DeepSeek Harness 工具。
DSH 负责阅读未知仓库、选择参数与修复动作；Python 核心负责执行并返回可审计证据。
插件不包含 ResNet、ViT、SuperRes 等模型专用适配器。

## 开发与测试

要求 Node.js 22、可调用的 DSH，以及已安装或可通过 `PYTHONPATH` 导入的 TorchPort。

```bash
cd integrations/dsh
npm install
npm test
npm run pack:check
```

`npm test` 会先编译 TypeScript，再运行分析循环和证书归一化测试。

## 安装到 DSH

```bash
dsh plugin --profile headless add "$PWD"

export TORCHPORT_PYTHON=/path/to/python
export TORCHPORT_PYTHONPATH=/path/to/torchport/src
export TORCHPORT_TRACE_DIR=/path/to/traces
```

包内 `dsh.bundle.patch` 会自动插入 `torchport` 插件行，无需手工编辑 DSH profile。
`DEEPSEEK_API_KEY` 只应通过进程环境或 DSH 凭据系统提供，不得写入仓库。

## 结构化工具

| 工具 | 职责 |
|---|---|
| `scan_repository` | 扫描框架、入口、依赖、权重和风险 |
| `create_migration_plan` | 生成不修改源码的迁移计划 |
| `start_migration` | 执行初始迁移、运行检查和可选对齐 |
| `run_verification` | 独立执行 PyTorch/Jittor 双端验证 |
| `locate_first_divergence` | 返回当前状态和首次数值分歧点 |
| `propose_repairs` | 把失败证据归约为下一轮候选动作 |
| `apply_verified_repair` | 记录 DSH 决策后重新迁移和验证 |
| `generate_certification_report` | 生成稳定、可发布的 DSH 报告 |

`modelArgs`、`modelKw` 和多输入描述使用真实 JSON schema，整数、布尔值和嵌套配置
不会退化成字符串。外部命令以参数数组启动，不经过 shell；完整报告写入输出目录，送回
LLM 的结果只保留证书、失败原因、入口链和下一步动作。

## 闭环

```text
扫描 → 规划 → 迁移 → 双端运行 → 数值证书
                         ↓ 失败
                首次分歧与结构化动作
                         ↓
                   DSH 选择修复
                         ↓
                    重新迁移验证
```

每次 Python 工具调用都写入 JSONL 轨迹；`auto.json` 保存 `agent.round`、
`agent.decision` 和 `agent.candidate`。FP32 通过即签发迁移证书，Strict 结果继续保留为
中间层诊断，不会被隐藏。

## 已验证端到端样本

在 PyTorch 官方 examples 的 `super_resolution` 子仓库上，真实 DSH 完成了扫描、
参数修复、Jittor 环境归因、重新验证和报告生成。最终结果为：

```text
status   verified
passed   true
FP32     true
Strict   false
first    conv3
```

复现实验摘要和脱敏证书位于 `experiments/stage3_method/` 与 `experiments/results/`。
