# SimCSE train-step v2：`ustat` 规则离线校准（2026-10-29）

状态：**未通过校准；没有更改验收代码或旧结果。** 本文只重判已经保存的
`simcse-train-reference-v2-r1` 和 `-repeat2` 输出，不使用 GPU。原隐藏
`verify` 的 `PARITY_FAILED`、`full-r4` 和 `none-r4` 的失败记录均保留。

## 证据和规则

- 同一 v2 TaskSpec digest：`ee6524fb32ad16de6358816f5a33a2232bdd96d9b1822cd3f7d9c0694260efb6`。
- source oracle：`D:\科研\SimCSE-Benchmark-v1\controller\train-reference-v2\task-datafs.json`
  对应的远端 `harness/7f8a3cab794f/evidence/oracle/ee6524fb32ad16de/`。
- 两次 target NPZ 的精确路径见控制端
  `train-v2-rule-calibration-20261029/ustat-sensitivity.json`；原始 NPZ 仍在远端。
- 离线脚本为本目录的 `calibrate_train_ustat.py`。旧规则与两种提议均沿用
  `k=8`、`floor=1e-7`，按每个输出张量的 max_abs 和 p99 判定。
  `drop_sum` 只比较 `ustat` 的 L2 范数和 max_abs；
  `sum_sketch_scale` 另外检查 sum，但把 sum 的 source 噪声基数设为
  `max(|source sum - fp64 sum|, |source sum - alt sum|,
  ||source sketch - fp64 sketch||₂, ||source sketch - alt sketch||₂)`。
  其余张量不变。表中的“最大超限倍数”是误差除以 `8 × 噪声 + floor`，
  大于 1 即失败；它不是原报告的 `ratio_max` 字段。

| target 运行 | 旧规则失败张量 | 去 sum 后失败 | 去 sum 最大超限倍数 | sketch 尺度后失败 | sketch 尺度最大超限倍数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2-r1 | 104 | 60 | 7.82 | 60 | 7.82 |
| v2-r1-repeat2 | 88 | 5 | 3.39 | 5 | 3.39 |

两种提议都**不能**让这两次参考运行通过，故没有正的通过余量。
v2-r1 去 sum 后最严重的是 `output.loss.13`；其余超限还包括多个
`output.ustat` 的 L2/max_abs 分量。repeat2 也是 loss 和 L2/max_abs
仍超限。仅修正 sum 不能解决这次校准，因此先补参考重复性证据；
新协议实现仍未开始。

进一步的**诊断提议**是：仅用控制端两次参考运行 r1/repeat2 的
逐张量重复差建立固定目标环境噪声基数，与 source FP32/FP64/alt
噪声取较大值；仍去掉 `ustat` sum，保持 k=8、floor=1e-7。
该基数在检查新候选前固定，不能从待判候选本身计算。
`analyze_repeat_envelope.py` 对全部 6,384 个输出张量重算：
r1/repeat2 的最大阈值使用率为 0.735；新增的三个有效参考
repeat3/4/6 作为留出运行，最大为 0.867，均无超限，最小余量约
13.3%。原始诊断 JSON 在控制端
`train-v2-rule-calibration-20261029/repeat-envelope-all.json`。
这是两次参考定义的噪声基数；下文记录负对照和新 pilot
TaskSpec 的隐藏 `verify`。标定样本量仍不足以冻结正式协议。

新增参考 repeat3/4/6 在原协议下仍 `PARITY_FAILED`，分别有
157/153/142 个超限张量；repeat5 在 Jittor 初始化时发生
`double free or corruption (fasttop)` 并 `RUN_FAILED`，没有数值输出，
不计入上表留出检验。各控制目录和远端原始 stderr/NPZ 均保留。

第一次名为 `simcse-train-neg-v2-adamw-bias-r0` 的尝试也**无效**：
旧控制脚本只同步了负对照目录，未把 TaskSpec 的 `target.repo`
绑定到该目录；候选身份与参考完全相同。它的输出不能用于判断
负对照检出率。`verify_train_reference.py` 已改为在独立控制目录
生成绑定后的 TaskSpec，并先验证 oracle digest 不变、target
identity 改变；真正的负对照另用新运行编号。

## 两个有效 v2 负对照

控制端 `simcse-train-neg-v2-adamw-bias-r1` 与
`simcse-train-neg-v2-temp-006-r1` 的候选身份分别为
`dc4da32324a38704d66618e20a52d5ed705ae082f8ea2c1b74cc8f9119ad81f0`、
`d943ad0911b655a38ed4a12ffa2c0f891b8197819dd305aeacd8ed69f3806a46`，
均不同于参考的 `31ac173abe600e5bd92ceeba3552ad9de47caacd2d7dc7eb1b28a69c24537253`。
两次结构和 199 个权重的完整性检查均通过；各自重新生成的
`outputs.npz`、`train_fp64.npz`、`train_alt.npz` 的 SHA256 与原
source oracle 三份文件分别完全相同，task digest 也相同。
旧 `verify` 仍按旧规则记录 `PARITY_FAILED`，不改写。

用上节固定参考噪声基数，对全部 6,384 个输出张量离线重判：

| 显式变异 | 超限张量 | 最坏阈值使用率 | 第 0 步 loss |
| --- | ---: | ---: | --- |
| AdamW 第一矩偏差校正错移一步 | 5,999 | 257,216.85 | 未超限；从第 1 步起的 15 个 loss 超限 |
| 温度 0.05 → 0.06 | 5,814 | 41,105.08 | 超限 |

原始输出留在各自远端运行目录；完整逐张量重判 JSON 在控制端
`train-v2-rule-calibration-20261029/{adamw-bias,temp-006}-valid-fixed-peer.json`。
两项预设负对照均失败，因此固定参考重复基数这条**pilot 规则**
已通过本轮检出检查。它仍只有两次参考来估噪声、三次有效留出
参考，不能据此宣称误判率 ≤5% 或正式协议已经冻结。

## 诊断边界

- source 的 `train_alt.npz` 与主路径 6,384/6,384 个 `output.*`
  张量逐位相同。所谓 source 噪声包络实际上只有 FP32 与 FP64 一种距离，
  对训练统计量明显偏薄。不能为了让两次参考恰好通过，直接从这两次
  target 输出反推候选可用的噪声下限。
- 原规则下 v2-r1 的 104 个超限中，102 个是 `ustat`，2 个是 loss；
  repeat2 的 88 个超限中有 1 个 loss。`output.loss.13` 的 v2-r1
  target 与 fp64 差约 `1.68e-6`，source 与 fp64 差约 `1.44e-8`；
  `output.loss.3` 的 target 差约 `3.48e-6`，两次 target 相差约
  `3.58e-6`。这支持“现有单样本包络过薄”，不证明模型实现正确。
- 控制端 `error-scale.json` 列出 Adam 更新的累计学习率上界和逐张量
  target 重复差。后期 `ustat` 有 `1e-4` 到 `6e-4` 量级波动；
  这与更新路径和重复运行波动相容，**尚未定位到特定算子或反向传播**。
- `concentration.json` 对原规则下 186 个 sum 失败项分析：两次
  target 的 sketch 差向量余弦相似度中位数约 `0.419`，前 10 个桶
  的能量占比中位数约 `0.703`。偏差并非纯粹均匀噪声；这也是不能
  无条件放大 sum 容差的理由。

本轮已取得与候选独立的重复性证据，并留出三个有效参考运行。
下一轮仍需扩充参考样本，估计误拒率。v1 控制目录中名为
`adamw-bias`、`temp-006` 的三份文件
与 v1 参考逐字相同，未编码负向变异，不能把目录名当作已完成的
负对照。已另在控制端 `train-negative-controls-v2/` 建立显式变异：
前者把 AdamW 第一矩偏差校正错移一步，后者把温度设为 0.06；
两者已做静态 diff、语法检查和独立 GPU 检验。

## 新 pilot 判定路径的端到端检查

已先在 `AGENTS.md` 写明训练验收现状，再将候选无关的固定重复性
基数接入 `compare_noise_calibrated`，命名
`train-repeat-envelope-v1-pilot`。基数是控制端 r1/repeat2 的逐张量
max_abs/p99 距离，NPZ SHA256 为
`c22e70b5e0d59fde6aafda0dd8ad9d8481732b7adef19f683e72521739ed0c3f`；
评判包把它放在 `judge/`，不放入 Agent 候选工作区。
训练 TaskSpec 加上名称、路径和哈希后，source oracle digest 仍是
`ee6524fb32ad16de6358816f5a33a2232bdd96d9b1822cd3f7d9c0694260efb6`。

控制端新运行 `simcse-train-repeat-envelope-reference-r0` 用该 TaskSpec
调用隐藏 `verify`：两次目标运行均 `passed=true`，各比较 6,384 个
输出张量、0 个超限；总报告 `passed=true`、`status=verified`，
`physical_cost.executions=4`、CUDA 执行墙钟 362.27 秒。
原始 `verify-result.json` 和远端 NPZ/manifest 保留在独立控制目录，
不回填 r4 或旧参考结果。此处“通过”只指新的 **pilot** 合约；
两次重复运行不能证明正式误拒率。

控制端旧分析脚本目前保留原位（`summarize_verify.py`、
`analyze_error_scale.py`、`analyze_thin_envelope.py`、
`analyze_zero_grad.py`、`analyze_concentration.py`）。它们固定了旧路径
和旧规则，作为原证据的生成方法留存；本目录新脚本通过参数读取
证据，可在仓库中复用。待新规则稳定后再决定是否把旧脚本整理成
通用的仓库分析工具，避免把只适用于旧控制目录的脚本误当成正式判定。

无效记录保留：`simcse-train-reference-v2-r0`（MSYS 路径错误）、
`simcse-train-reference-v2-r1-repeat`（内存回收中断）和
`train-reference-v2/task.json`（未用于上述远端 verify）。它们不计入
参考通过率，也不删除。
