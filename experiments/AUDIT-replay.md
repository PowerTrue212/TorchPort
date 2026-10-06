# 回放污染审计（P0）

修复前，通用 capture 会把源端 npz 整包交给目标端 `OpTrace`，同名 `op.*` 直接返回源端值，目标算子不执行。

| 结果 | 路径 | 状态 |
| --- | --- | --- |
| `experiments/data/current.json` 中 8 个 `dsh_torchport` verified（pytorch_unet、resnet_cifar10、senet、mae、superres、swin、vit_pytorch、vae） | 通用 align/capture | 可能受影响，需重跑 |
| 其中 mae（`fix-random-replay`） | 依赖 rand 回放通过 | 高风险，重跑时须按受控随机性单独标注 |
| `experiments/out/cases.csv`、`case_table.md`、`figures/fig_vit_mae/` | 由 current.json 派生 | 随上游重跑重新生成 |
| `official_jittor` 各行 | 未经过 capture | 不受影响 |
| `experiments/results/raft-v1-2026-08-25-v100/` | 专用 evaluator | 不受影响 |

原始文件保留不改。重跑后在此表补充新运行目录。
