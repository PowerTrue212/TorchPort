# LoRA GPT-2 Small r1 四指标核对（2026-10-25）

只复用 r1 的冻结候选和现有合成 checkpoint；本次未训练新模型，也未把缺项记成 0。
这份记录是试跑诊断，不把 `valid=false` 的两组追认为正式对照实验。

| 组别 | R_e | B_ℓ 正式值 | B_ℓ 探索值：自然轨迹 / 隔离执行 | P_t | E_m |
| --- | ---: | --- | ---: | --- | --- |
| full | 1 | 未定义 | 63.3278 / 97.7499 | 未定义 | 未定义 |
| none | 1 | 未定义 | 63.8309 / 96.4341 | 未定义 | 未定义 |

`R_e=1` 的依据是两份冻结快照在 V100 隐藏验收中均完成固定 batch 前向和连续
四步训练，且两项各重复 2/2 通过。它只表示入口可运行；控制端两组仍是
`passed=true, valid=false, success=false`。原结论和污染证据见
[r1 试跑结案](r1-pilot-outcome-20261024.md)。

## B_ℓ 的探索性计算

新增 trace 使用与原隐藏前向相同的 source、合成权重和固定输入，目标分别指向
full/none 的冻结候选。源码权重所属模块给出 76 个检查点；`attn.bias` 是因果
mask buffer，不算拥有参数的模块。按 [Benchmark v1 协议](../../../docs/benchmark-v1-protocol.md)
的现行公式，缺失、形状不符和非有限检查点得 0；其余点使用代码中已有的
`operator-strict` 容差 `a=1e-6, r=1e-5`。逐点 `d` 和得分在下列 JSON；
[计算脚本](score_r1_layers.py) 可从原始 NPZ 重新算出。

| 组别 | 轨迹 | 得分 | `d>1` 点数 | 缺失/形状错误点数 |
| --- | --- | ---: | ---: | ---: |
| full | 自然轨迹 | 63.3278 | 45 | 1 |
| full | 隔离执行 | 97.7499 | 8 | 0 |
| none | 自然轨迹 | 63.8309 | 46 | 2 |
| none | 隔离执行 | 96.4341 | 8 | 1 |

自然轨迹里，两组的 `model.transformer.wpe` 输出形状是 source `(2,16,768)`、
target `(1,16,768)`，因此按公式该点为 0；隔离执行喂入相同局部输入后两组
的形状一致。none 的 `model.lm_head.decoder` 不在候选 trace 中，自然轨迹和
隔离执行都将该点记 0。不能把检查点缺失直接解释成最终 logits 错误：两组
最终前向验收仍通过。自然轨迹含上游误差累积，隔离执行则用于区分局部行为。

正式 `B_ℓ` 仍未定义：r1 启动前没有把这 76 个检查点和逐层容差固定进 TaskSpec；
此次是事后选择。trace 自身的严格比较返回 `status=diverged`，它是诊断结果，
不覆盖原先通过的隐藏 `verify`。两种轨迹的分数也不能混作同一个指标。

## 不能计算的两项

- `P_t` 需要任务训练后的 GPT-2 Small LoRA checkpoint，以及相同测试集、
  预处理、生成/评分代码和事先定好的容差。r1 只有 base 加合成 LoRA 权重；
  固定 batch logits 和四步训练不是任务性能。没有新训练，故未定义。
- `E_m` 公式依赖任务成功 `s` 与可比较的 Agent 有效时间、Token、工具调用和
  失败候选数。`P_t` 缺失使 `s` 无法确定；full 开发不在控制端记录内，none
  曾读取 full 候选与报告。不能用补验进程的几秒或 `oracle_queries=0` 代替
  Agent 成本，故未定义。

## 原始证据与复算

控制端目录 `D:\科研\LoRA-Benchmark-v1\metrics-r1\` 保存：

- `full/trial/remote-raw/0001-trace.json` 和 `none/trial/remote-raw/0001-trace.json`：
  远端 trace 原始报告；两组 TaskSpec、gateway 状态与事件也在各自目录。
- `full-source-module-outputs.npz`、`none-source-module-outputs.npz`：source
  模块输出；`full-target-trace.npz`、`none-target-trace.npz`：自然轨迹；
  `full-target-isolated.npz`、`none-target-isolated.npz`：隔离执行结果。
- `full-layer-natural.json`、`full-layer-isolated.json`、`none-layer-natural.json`、
  `none-layer-isolated.json`：76 点逐项 `d`、得分和缺失原因。原始隐藏验收
  JSON 位于 `D:\科研\LoRA-Benchmark-v1\controller\full-r1-pilot\trial\` 和
  `...\none-r1-frozen-recert-v2\trial\`。

本次只删去与新 full source NPZ SHA256 完全相同的一份中断重跑副本；上述原始
JSON 和必需 NPZ 均保留。远端 Jittor 缓存未纳入长期证据。
