# LightGCN LastFM 训练后任务指标（3080 Ti，pilot）

状态：**冻结源端训练的 checkpoint 已生成；Jittor 控制参考以同一完整权重和官方 LastFM 测试集重放，Top-20 排序与 PyTorch 全部相同。** 这补上两步随机初始化训练校准缺少的任务指标，但仍不是 Benchmark v1 的正式 Agent trial。

## 固定材料和训练

- 源码：作者 `LightGCN-PyTorch` 快照 `947ca2b3`，源码包 SHA256 `7b21fc57dfbbfb1a74e16e729b0ef45da878af54fdf2d18cb10fca262275e948`。LastFM `data1.txt` / `test1.txt` SHA256 分别为 `c82af3228cbbda2f55a39244de09ed93292894f1d0d9f1908ebb57e27598fded`、`de4417c0e6123284ce59a0934bcf5288ec1ec8774a18a1eef453c9952dfa48f9`。
- 在独立数据盘目录重解源码包，调用作者 `code/main.py` 训练 10 epoch：3 层、64 维、BPR batch 2048、Adam lr 0.001、正则 1e-4、seed 20261102、dropout 0。包装器显式关闭 TF32，启用确定性算法和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。源端为 PyTorch 2.4.1+cu121；额外依赖装在仓库独立 `deps/`。
- 训练日志 loss 从 epoch 1 的约 0.689 降到 epoch 10 的约 0.412。完整 checkpoint `lgn-lastfm-3-64.pth.tar` SHA256 `cb919a7a2a6edf6b33db08e3770dd7119c56d3cd77bb0d2a7624d4f7c85a431c`，两项 embedding 权重完整加载。它是本轮按固定源码、数据、seed 训练出的 checkpoint，并非论文作者发布的预训练权重。

## 真实测试集指标与目标重放

作者的 `Procedure.Test` 在 1,858 个测试用户上评估，排除各用户训练集正样本，得到 `Precision@20=0.0329117330`、`Recall@20=0.1195417746`、`NDCG@20=0.0813804413`。训练前 epoch 0 的日志分别约 `0.00164155`、`0.0061529`、`0.00353987`。单独导出的源端 Top-20 和评分复算出与作者评估函数一致的指标（误差 `<1e-12`），因此后续跨框架比较使用同一用户顺序与排除规则。

将 checkpoint 的全部两张 embedding 表导出为 `weights.npz`（SHA256 `8a42b8bdfd96262faf094126daaba210762515101594813def00f4089956af60`），在 Jittor 1.3.11.0 / CUDA 12.8 / `conv_opt=1` 的控制参考中逐参数回读核验。目标端按源端 `getUsersRating` 执行 Sigmoid，再排除训练集正样本：

| 运行 | Precision@20 | Recall@20 | NDCG@20 | 与源端逐用户 Top-20 |
| --- | ---: | ---: | ---: | ---: |
| PyTorch source | 0.0329117330 | 0.1195417746 | 0.0813804413 | 基准 |
| Jittor r2 | 0.0329117330 | 0.1195417746 | 0.0813804413 | 1858/1858 完全相同 |
| Jittor r3 重复 | 0.0329117330 | 0.1195417746 | 0.0813804413 | 1858/1858 完全相同 |
| 负对照：仅用 2 层传播 | 0.0345263724 | 0.1250616140 | 0.0847491808 | 0/1858 完全相同 |

Jittor r2 与源端全部 8,340,562 个评分（含同值的训练项屏蔽位）最大绝对差 `1.7881393432617188e-7`，p99 `5.960464477539063e-8`；r2/r3 的评分和 Top-20 文件 SHA256 分别逐位相同。2 层负对照的指标反而更高，但它与源端只在约 32.95% 的 Top-20 槽位一致。这证明单看推荐指标不足以验收迁移语义；数值与排序对齐仍要单独检查。

首次目标重放 `target-r1` 漏掉了作者 `getUsersRating` 中的 Sigmoid；其原始评分差很大但 Top-20 与指标不变。已在 `target-r2` 修正，保留 `target-r1` 原始记录作为无效预检，不纳入上述目标数值结论。

## 证据与边界

源训练日志、协议 JSON、checkpoint、源/目标完整评分和 Top-20 NPY、逐项 SHA256、负对照及两次目标重放均在 `D:\科研\LightGCN-Benchmark-v1\controller\task-metrics-3080ti-r2.tar.gz`；远端与本地归档 SHA256 均为 `c6b6561bf9e2824004da95f7430b428dd0125a31a28bb5379c2b66796449c633`。两步训练校准的独立证据见 `3080ti-pilot-20261101.md`。

本轮只训练一个固定 seed、10 epoch，并用一个控制端 Jittor 参考重放。没有重复完整训练或估计推荐指标的运行间波动。checkpoint 后续已进入[独立的五步训练隐藏 pilot TaskSpec](train-ckpt-five-step-3080ti-20261102.md)，但它从新 Adam 状态 warm start，不是原训练进程续训。结论为 **pilot_only**；正式 Agent trial 仍待 GATE、预算和停止条件冻结。
