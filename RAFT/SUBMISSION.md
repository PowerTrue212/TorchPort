# RAFT 实验提交清单

## 应提交

```text
.gitignore
RAFT/                              # 协议、prompt、公共 evaluator、报告脚本和测试
experiments/README.md              # 原始层 / 公开证据层的数据规则
experiments/scripts/export_raft.py # 从 data-fs 生成脱敏公开 bundle
experiments/scripts/extract_raft_features.py # 从真实双端 trace 提取紧凑 feature 证据
experiments/results/raft-v1-2026-08-25-v100/
experiments/figures/fig_raft/
experiments/data/tensors/raft_features.npz
experiments/data/tensors/raft_features_meta.json
```

推荐命令：

```powershell
git add .gitignore RAFT `
  experiments/README.md `
  experiments/scripts/export_raft.py experiments/scripts/extract_raft_features.py `
  experiments/results/raft-v1-2026-08-25-v100 `
  experiments/figures/fig_raft `
  experiments/data/tensors/raft_features.npz `
  experiments/data/tensors/raft_features_meta.json
```

提交前验证：

```powershell
python RAFT/scripts/build_report.py
python RAFT/scripts/validate_protocol.py --stage results
python -m pytest RAFT/tests/test_protocol.py -q
python experiments/scripts/export_raft.py
python experiments/scripts/extract_raft_features.py
python experiments/figures/fig_raft/plot.py
git diff --cached --stat
```

## 不应提交

```text
RAFT/source/
RAFT/shared/                       # checkpoint、fixture、reference、weights
RAFT/workspaces/                   # 三方法迁移工作区
RAFT/results/                      # flow NPY、attempts、完整运行 trace
RAFT/outputs/                      # data-fs 内的可再生成输出
.tmp/、.torchport/、.trash/
V100_CONNECTION.md、SSH key、主机地址和会话原文
根目录临时 probe/debug/submission 文件
```

特别说明：`RAFT/results/dsh_torchport/traces/source.npz` 和 `target.npz` 是约 31 MB
的原始双端捕获，只保留在 data-fs/本地忽略目录；提交的是由其确定性导出的紧凑
`raft_features.npz`、元数据、CSV 和图片。

公开结果的事实来源是
`experiments/results/raft-v1-2026-08-25-v100/manifest.json`。大文件只在 V100/data-fs
保留，并由逻辑路径、SHA256 和字节数引用。

## 当前结果口径

- Codex：`passed`，RN=true，NA=true，Rounds=5。
- pure DSH：`stopped_failed`，RN=true，NA=false，Rounds=1；不能写成 forward 崩溃。
- DSH + TorchPort：公共 FP32 evaluator `passed`，RN=true，NA=true，Rounds=1；
  TorchPort strict 仍为 false，必须在补充报告中如实保留。
- Agent elapsed/token 只在三种方法均实际采集后再进入对比图；本次图表不显示未完整
  记录的成本字段，也不按 0 处理。
