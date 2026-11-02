# LightGCN 已训练权重五步检验（3080 Ti，pilot）

状态：**五步真实 LastFM batch 的训练数值检验通过；11 次参考留一检验零误拒，三种显式负对照全部检出；两次新增隐藏 pilot `verify` 均通过。** 这仍不是 Benchmark v1 正式 Agent trial。

## 冻结任务

- 初始参数来自 [任务指标实验](task-metrics-3080ti-20261102.md)中固定训练 10 epoch 的完整 PyTorch checkpoint，导出权重 SHA256 `8a42b8bdfd96262faf094126daaba210762515101594813def00f4089956af60`。两端各加载两张 embedding 表并回读核验。此 TaskSpec 只加载参数，Adam 动量从零开始；它是已训练参数的 **warm start**，不是原训练进程的优化器状态续训。
- 输入来自冻结 LastFM `data1.txt`（SHA256 `c82af3228cbbda2f55a39244de09ed93292894f1d0d9f1908ebb57e27598fded`）：seed `20261102` 无放回选 20 条训练正样本，按用户真实交互集合排除后为每条抽一个负样本，组成 5 个不同的 batch、每批 4 例。三个输入 NPY SHA256 分别为 `266a64efb6b3cdbc1418f7bcc2014643a1d1841cc067b73e5582f06472e8ed25`、`a1079c11191ab7bb8f19f1f57c68c91c4bccb94346b72b542f7ce7dbe0d5954f`、`b6d7d3ed75210ebd34f4e80676dccda553c938100d30439a4fea79170b35b498`。
- 架构仍为 3 层、64 维，BPR+1e-4 正则，Adam lr 0.001，dropout 0；环境仍为 PyTorch 2.4.1+cu121 / Jittor 1.3.11.0 / RTX 3080 Ti，source 显式关闭 TF32、开启确定性算法并设置 cuBLAS 工作区，target 设 `conv_opt=1`。每步比较 loss、学习率和两张参数表的完整更新 sketch（4096 桶）。任务 oracle digest `cbbc8f7cc516eb7ed633c528bece83316265045248a079f78564482972f63008`，不同于先前随机权重两步任务，source oracle、FP64 和 alt 均重新建立。
- 作为 warm-start 预检，同一训练 checkpoint 与旧两步固定输入构成另一个独立任务：新 source oracle digest `6534d11af0ac71c84d354342440f70a5c27d8c0fe04b87cd5da695b6f088cece`，正确参考按现行规则 10/10 零超限，学习率和正则负对照分别有 9/10、2/10 超限。原始证据独立归档 `controller/train-ckpt-two-3080ti-r1-evidence.tar.gz`，SHA256 `9022d90e47357576135bae6a1c012bb82eb46f0acafc6c2c9e377d9c9b16275e`；不与五步任务合并计算校准率。

## 验收与校准

| 条件 | 现行规则 `verify` | 新 pilot 规则 `verify` |
| --- | --- | --- |
| 正确 Jittor 控制参考 | 通过，25/25 张量零超限 | 两次新增隐藏复验均通过，25/25 零超限 |
| Adam 学习率 ×1.01 | `PARITY_FAILED`，24/25 超限 | `PARITY_FAILED`，24/25 超限 |
| BPR 正则 ×1.01 | `PARITY_FAILED`，14/25 超限 | `PARITY_FAILED`，10/25 超限 |
| 图传播层数 3→2 | `PARITY_FAILED`，25/25 超限 | `PARITY_FAILED`，25/25 超限 |

包含 oracle 在内的 source FP32 共 11 次，25 个输出中 20 个跨运行不同，最大绝对差 `7.000553523539566e-7`；每项已观察到的 source peer 差仍低于该项 FP32/FP64/alt 标定差。控制端参考共 11 次（首次隐藏 `verify` 加 10 次独立运行）。按同一定义建立只在评判侧使用的目标重复基数，SHA256 `d88be0a61fcb54173e23d484f7182fdf05217fb574cd4257a91f3e6b3e657240`；留一检验 11/11 零误拒，最差阈值使用率约 0.494。三种负对照在同一固定规则下分别有 24、10、25 个超限张量。新增两次正确参考的隐藏 pilot `verify` 均通过。

## 证据与边界

原始 task、输入、oracle FP32/FP64/alt NPZ、全部 source/target 运行、三种负对照与两次新增隐藏复验 JSON/NPZ/日志在 `D:\科研\LightGCN-Benchmark-v1\controller\train-ckpt-five-3080ti-r2-evidence.tar.gz`；远端与本地归档 SHA256 均为 `4ca7f6f0e91085ff62d7c1980dba92f3e36ea9cfb592bd6935af43c7382b828a`。校准文件和留一结果在本地 `controller/train-ckpt-five-3080ti-r1-calibration-r1/`。旧两步任务与 V100 判定保持原样。

仍未知：只用一个固定训练 checkpoint、5 个 batch×4 例、一个正确目标实现；没有真实 optimizer 动量续训、更多 seed/数据切片，也未冻结正式预算和停止条件。任务级推荐指标另由同 checkpoint 的推理重放证明，但不替代五步训练数值检验。当前标记 **pilot_only**；正式 Agent trial 仍不可启动。
