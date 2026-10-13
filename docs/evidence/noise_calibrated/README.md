# Noise-calibrated 模块隔离验收规则：证据汇编

更新时间：2026-10-xx（本证据包生成时间，见 `summary.json` 内各原始 JSON 的时间戳）
状态：**诊断证据，不是正式 verify 协议**。本文档的所有判定都来自 `trace`/`localize` 产生的
隔离证据，不改变、不覆盖任何冻结 `verify` 任务的失败结果（见“与冻结协议的关系”一节）。

## 1. 结论先行

1. 把 `src/torchport/comparison/envelope.py::compare_noise_calibrated` 的判定统一按 **k=8、全部模块
   gating**（不是叶子模块、不是探索期的 k=4）重新计算后：
   - 两个正确权重配置（3080 Ti 随机权重、3080 Ti 真实 PVTv2 encoder 权重）和两台 GPU
     （3080 Ti、V100）上的随机权重基线全部通过，最大 ratio 分别为 **1.836**、**3.894**、
     **3.167**。
   - 四个"算子语义突变"负对照里，**3 个有效**（`ln_eps`、`mask_residual`、`gelu_tanh`）
     在两台 GPU 上都被规则正确拒绝，最小 ratio 为 **25.974**（V100 ln_eps）；`l2_eps`
     在当前输入上没有数值效果，两台 GPU 都通过（ratio 恒为基线值，说明注入点在当前
     输入下未被触发，不是规则漏判）。
   - 分离边界：**max(correct) = 3.894 < k=8 < min(effective mutant) = 25.974**，安全系数
     约 6.67 倍（min_mutant / max_correct）。
2. **leaf-only gating 会漏判。** `mask_residual` 在只看 135 个叶子张量时 0 个失败
   （max ratio 仍是基线的 1.836，因为突变点在组合模块的输出，不在任何叶子模块的输出）；
   把全部 271 个模块（含 `model.decoder.MSA*`、`model.decoder.*`、`model.*` 等组合模块）
   纳入 gating 后，20 个张量失败，max ratio 跳到 **2.31e7**。这是本次校准把 gating 口径从
   叶子改成全模块的直接原因。
3. **固定容差无法替代 noise-calibrated 规则。** `isolated.strict`（operator-strict 隔离
   逐张量比较）和整网 `output`（fp32-inference 容差）在全部 13 组运行上都是 FAIL——
   包括两个已知正确的权重配置。固定容差对"正确迁移但后端数值路径不同"和"迁移语义
   错误"给出同一个判定，不能区分。整网固定容差尤其不能区分 `ln_eps`：mismatch 元素数
   535（ln_eps）、569（l2_eps，当前输入下无效的负对照）、574（correct_random 基线）
   三者几乎相同，三者的 ratio 却是 26.98（有效拒绝）、1.84（基线量级）、1.84（基线量级）——
   只有 noise-calibrated 的 ratio 把它们分开。
4. **`conv_opt=0` 不是"更干净的"同义后端配置，而是本身会触发规则失败的变体。**
   它不是本报告的负对照（没有改模型语义），但在 3080 Ti 上重新判定后有 12/271 个模块
   失败、max ratio 248.3，远超 k=8，且整网固定容差失败元素从 574 升到 57,050。这与
   AGENTS.md 记录的"整网更差"一致，这里补充了模块级证据：`conv_opt=0` 改变了目标端
   后端数值路径本身，不能作为"验证 target 后端无关性"的对照，只能作为"target 后端
   配置敏感性"的证据。
5. **这套规则目前只在一个真实迁移（CamoFormer）、一个候选实现、2 种权重设置、
   2 类 GPU、1 个 Jittor 版本、4 个负对照（3 个有效）上标定过。** k=8 是从这些数据里
   手工选的探索参数，不是独立标定或理论推导的值；用同一批数据选参数再用同一批数据
   证明参数有效，本身是循环论证。详见第 6 节“威胁有效性”。

## 2. 规则定义

```
对每个模块张量 T：
    noise      = max(err(source_fp32, source_fp64), err(source_fp32, source_alt))
    candidate  = err(target_fp32, source_fp64)
    通过条件：candidate_max_abs <= k * noise_max_abs + atol_floor
          AND candidate_p99_abs <= k * noise_p99_abs + atol_floor
    k 默认 8.0，atol_floor 默认 1e-7
    ratio_max = candidate_max_abs / max(noise_max_abs, atol_floor)
    ratio_p99 = candidate_p99_abs / max(noise_p99_abs, atol_floor)
```

来源：`src/torchport/comparison/envelope.py` 的 `compare_noise_calibrated`（本次只读取既有 trace 产物，
未修改该文件）。`source_alt` 是 source 端另一条独立 fp32 执行路径（例如重复执行一次），
用于估计 source 自身在 fp32 下的数值噪声；`fp64` 是同一输入下的 float64 执行结果，作为
高精度参考。`leaves` 参数控制 gating 范围：传入叶子模块集合时只对叶子模块的失败计入
`first_gating_divergence`／最终 `passed`；`None` 时所有张量都 gating。

**本报告统一传 `leaves=None`（全部模块 gating）、`k=8`，并且不信任任何原始 JSON 里存的
`k`、`gating`、`passed` 字段**，全部从每个张量的 `ratio_max`/`ratio_p99`/`candidate_*`/
`noise_*` 原始数值重新计算。原因：多数早期 3080 Ti trace 文件在生成时用的是探索期
`k=4`，且 `gating` 字段只标记了 135/271 个叶子模块；V100 的 trace 文件生成时已经用
`k=8`、全模块 gating，可以作为交叉核对（见第 4.1 节两者一致）。重新判定逻辑见
`build_evidence.py` 的 `rejudge_tensors`/`rejudged_summary`。

## 3. 数据来源与出处

| 文件（相对仓库根目录） | GPU / 后端 | 权重 | 用途 |
| --- | --- | --- | --- |
| `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/trace-module-isolation-calibrated-r3.json` | 3080 Ti, sm_86, source 关 TF32 | 随机权重 | 正确基线（随机权重） |
| `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/trace-encoder-ckpt-r1.json` | 3080 Ti, sm_86 | 真实 PVTv2-B4 encoder，decoder 随机（`complete=false`） | 正确基线（部分真实权重） |
| `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/trace-source-repeat-r1.json` | 3080 Ti, sm_86 | 随机权重 | 候选身份与 r3 相同，用于目标端重复性对照（见 `isolated.repeatability`） |
| `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/trace-convopt0-r1.json` | 3080 Ti, sm_86, target `conv_opt=0` | 随机权重 | 后端变体（非语义突变），验证 target 后端配置敏感性 |
| `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/negctl/trace-ln_eps.json` | 3080 Ti | 随机权重 | 负对照：LayerNorm eps 改变 |
| `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/negctl/trace-mask_residual.json` | 3080 Ti | 随机权重 | 负对照：mask 作用到整个残差而非仅 q/k |
| `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/negctl/trace-gelu_tanh.json` | 3080 Ti | 随机权重 | 负对照：GELU 用 tanh 近似替代 erf 精确式 |
| `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/negctl/trace-l2_eps.json` | 3080 Ti | 随机权重 | 负对照：L2 归一化 eps 改变（当前输入下无数值效果） |
| `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/trace-v100-baseline-r6.json` | V100, sm_70, source 关 TF32 | 随机权重 | 正确基线（第二类 GPU 交叉验证） |
| `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/negctl/trace-v100-{ln_eps,mask_residual,gelu_tanh,l2_eps}.json` | V100 | 随机权重 | 同一组负对照在 V100 上复现 |
| `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/v100-report.md` | V100 | — | 人工整理的 V100 对照摘要，用于交叉核对本报告的重判结果 |
| `migrations/CamoFormer/outputs/task-metrics/metrics-torch.json`、`metrics-jittor.json` | 3080 Ti（任务级） | **真实训练 checkpoint**（`CamoFormer-trained`） | 任务级指标（MAE/mean_E/S-measure/weighted-F），CAMO/CHAMELEON/COD10K/NC4K 四个真实数据集 |
| `migrations/CamoFormer/outputs/task-metrics/torch-vs-jittor-pixel-comparison.json` | 3080 Ti（任务级） | 同上 | 逐图像素级差异（0-255 量化后的预测图） |

权重设置混用提醒：模块隔离的"正确基线"用的是随机权重或随机权重+部分真实 encoder；
任务级指标用的是**完整真实训练 checkpoint**。两组实验不是同一个权重设置下的对照，
不能把"模块隔离通过"直接当作"任务级指标也会通过"的证明；两者只是同一候选实现
在不同验收口径下分别产生的独立证据。

## 4. 重判结果表（k=8，全模块 gating，数字取自 `summary.json.rejudged_table`）

### 4.1 全部 13 组运行

| GPU | run | 类型 | max(ratio_max,ratio_p99) | 失败张量数/271 | noise-calibrated 通过 | isolated.strict 通过 | 整网 fp32-inference 通过 | 整网失败元素/131072 |
| --- | --- | --- | ---: | ---: | :---: | :---: | :---: | ---: |
| 3080 Ti | correct_random | 正确基线 | 1.836 | 0 | PASS | FAIL | FAIL | 574 |
| 3080 Ti | correct_encoder_ckpt | 正确基线 | 3.894 | 0 | PASS | FAIL | FAIL | 38 |
| 3080 Ti | source_repeat | 正确基线（同候选） | 1.836 | 0 | PASS | FAIL | FAIL | 575 |
| 3080 Ti | convopt0 | 后端变体（非突变） | 248.286 | 12 | FAIL | FAIL | FAIL | 57,050 |
| 3080 Ti | mut_ln_eps | 有效负对照 | 26.975 | 12 | FAIL | FAIL | FAIL | 535 |
| 3080 Ti | mut_mask_residual | 有效负对照 | 2.312e7 | 20 | FAIL | FAIL | FAIL | 65,534 |
| 3080 Ti | mut_gelu_tanh | 有效负对照 | 264.924 | 35 | FAIL | FAIL | FAIL | 28,231 |
| 3080 Ti | mut_l2_eps | 无效负对照 | 1.836 | 0 | PASS | FAIL | FAIL | 569 |
| V100 | correct_random | 正确基线 | 3.167 | 0 | PASS | FAIL | FAIL | 66,359 |
| V100 | mut_ln_eps | 有效负对照 | 25.974 | 15 | FAIL | FAIL | FAIL | 66,325 |
| V100 | mut_mask_residual | 有效负对照 | 2.287e7 | 20 | FAIL | FAIL | FAIL | 74,534 |
| V100 | mut_gelu_tanh | 有效负对照 | 692.695 | 26 | FAIL | FAIL | FAIL | 65,428 |
| V100 | mut_l2_eps | 无效负对照 | 3.226 | 0 | PASS | FAIL | FAIL | 66,373 |

与原始声称数字（AGENTS.md / 任务输入）的对照：

| 声称值 | 本报告重判值 | 来源 | 一致性 |
| --- | --- | --- | --- |
| 正确基线 ratio 1.84 | 1.8362836647902712（3080 Ti correct_random） | 文件内原始存储即为此值（当时 k=4、叶子 gating 下的 max_gating_ratio） | 一致，k=8/全模块重判不改变这个最大值（该模块本身是叶子，且其 ratio 恰好是全局最大） |
| 正确基线 ratio 3.89 | 3.8940492509130076（3080 Ti correct_encoder_ckpt） | 同上 | 一致 |
| （任务未列出但交叉核对用）3.17 | 3.1668911718252697（V100 correct_random） | `v100-report.md` 已用 k=8 全模块 gating 记录同一数字 | 一致 |
| ln_eps ≈ 27 | 26.975（3080 Ti）、25.974（V100） | 重判 | 基本一致；V100 略低于"约 27"但仍远超 k=8 |
| gelu ≈ 265 / 693 | 264.924（3080 Ti）、692.695（V100） | 重判 | 一致 |
| mask ≈ 2.3e7 | 2.312e7（3080 Ti）、2.287e7（V100） | 重判 | 一致 |
| l2_eps 当前输入无效果 | 两台 GPU 的 l2_eps ratio 都等于各自基线值（1.836 / 3.226），没有自己的峰值 | 重判 | 一致：`l2_eps` 不能算作已验证的有效负对照，只能证明"规则在无效突变上不会误报" |
| ln_eps 整网固定容差 535 vs 569 | 535（3080 Ti mut_ln_eps 整网 fp32-inference 失败元素）vs 569（3080 Ti mut_l2_eps 整网失败元素） | `output.metrics.mismatched_elements` | 一致：两者（以及基线的 574）在整网固定容差口径下几乎无法区分，印证"固定容差区分力不足"的结论 |

**全部一致，没有发现与任务描述数字矛盾的地方。**

### 4.2 leaf-only vs all-module gating（mask_residual，3080 Ti）

| gating 口径 | 张量数 | max ratio | k=8 下失败张量数 |
| --- | --- | --- | --- |
| 仅叶子模块（旧口径，135 个） | 135 | 1.836 | 0 |
| 全部模块（135 叶子 + 136 组合，271 个） | 271 | 2.312e7 | 20 |

失败的 20 个张量全部是组合模块：`model.decoder.MSA{2,3,4,5}.{F_TA,B_TA}`、
`model.decoder.MSA{2,3,4,5}`、`model.decoder.{1,2,3,4}`、`model.{1,2,3,4}`。叶子模块
（例如 `model.decoder.MSA5.F_TA.norm1.body`）本身的输出没有越界，因为 `mask_residual`
突变改变的是"掩码作用在哪一步"，影响体现在组合模块把多个子模块输出合并之后的张量，
不体现在任何单个叶子算子的输出上。**只看叶子模块会完全漏掉这个有效负对照**，这是本次
把 gating 范围从叶子改成全部模块的直接证据。

## 5. 任务级指标（真实 checkpoint，供交叉参考，不是模块隔离的一部分）

| 数据集 | 图片数 | 像素差异比例 fraction_pixels_differing | 平均绝对差（0-255 量化） | \|mae 差\| | \|mean_E 差\| | \|S-measure 差\| | \|weighted-F 差\| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CAMO | 250 | 2.757e-4 | 2.757e-4 | 6.40e-9 | 2.71e-8 | 7.78e-8 | 6.19e-8 |
| CHAMELEON | 76 | 1.794e-4 | 1.794e-4 | 7.14e-9 | 1.02e-7 | 8.67e-8 | 7.60e-8 |
| COD10K | 2026 | 1.506e-4 | 1.506e-4 | 3.68e-10 | 8.01e-9 | 9.38e-9 | 2.82e-8 |
| NC4K | 4121 | 1.856e-4 | 1.856e-4 | 9.83e-9 | 4.70e-8 | 4.93e-8 | 7.50e-8 |

逐图 `fraction_pixels_differing`（6473 张图合并）：均值 1.787e-4，中位数 1.281e-4，
最大值 1.672e-3，最小值 0。这组任务级指标差异量级远小于论文复现本身的偏差
（`deviation_from_paper` 在 1e-2 ~ 1e-3 量级），说明在**完整真实训练权重**下，候选实现
的 Jittor 输出与 PyTorch 参考在任务指标层面高度接近。但这组证据来自不同的权重设置
（真实 checkpoint，非随机权重），**不能直接作为模块隔离 noise-calibrated 规则本身在
k=8 下成立的证据**，只能作为同一候选实现的独立正面信号。

## 6. 分离边界

`max(correct ratio across all runs) = 3.894`（3080 Ti correct_encoder_ckpt）
`min(effective-mutant ratio across all runs) = 25.974`（V100 mut_ln_eps）

`separation_margin_factor = 25.974 / 3.894 ≈ 6.67`

在当前这批数据下，任何 `k ∈ (3.894, 25.974)` 都能把全部正确基线与全部有效负对照分开；
k=8 落在这个区间内，但区间上界主要由样本量很小的 4 个负对照（3 个有效）决定，边界本身
很不稳定——换一个突变、换一个模型，边界完全可能移动到 k=8 的另一侧。见图 (g)。

## 7. 图表

所有图表由 `build_evidence.py` 生成，位于 `figs/`。

- `figs/fig_a_ratio_distribution.png` —— 每个 run 的 271 个模块张量 `max(ratio_max,
  ratio_p99)` 散点分布（log y 轴），黑色三角标出该 run 的最大值，红色虚线是 k=8。
  主图：一眼看出 3 个有效负对照和 `conv_opt=0` 的最大值明显越过红线，其余运行都在
  红线以下。
- `figs/fig_b_verdict_matrix.png` —— 固定容差（`isolated.strict` operator-strict 隔离、
  `output` fp32-inference 整网）与 noise-calibrated（k=8 重判）三栏判定矩阵，13 行全列。
  两个固定容差栏全部是 FAIL（包括正确基线），只有第三栏能把正确基线和突变分开。
- `figs/fig_c_leaf_vs_allmodule.png` —— `mask_residual` 负对照下 leaf-only vs all-module
  gating 的 ratio 箱线图对比，直观展示 leaf-only 完全漏判（左图全部在 k=8 以下），
  all-module 正确检出（右图 20 个点在 1e5~1e7 量级）。
- `figs/fig_d_ratio_vs_depth.png` —— ratio 随模块在 trace 中位置（近似深度/执行顺序）的
  变化，两个正确基线（随机权重、真实 encoder 权重）与 `mut_ln_eps` 的对比。可以看到
  `ln_eps` 的异常集中在 decoder 的 MSA 模块区域（position ≈ 170-200），编码器区域
  （position < 170）与正确基线几乎重合。
- `figs/fig_e_repeatability.png` —— source 自重复与 target 自重复在 `operator-strict`
  容差下的 mismatch 元素数与最大绝对差，所有有记录的 run。source 全部逐位一致
  （mismatch=0），target 普遍有数千到数万元素量级的 run-to-run 差异，`conv_opt=0` 的
  target 自重复差异（72,158 个元素，max_abs=0.117）明显比其余 run 更大。
- `figs/fig_f_task_level.png` —— 左：四个数据集的任务指标绝对差（log y），右：逐图像素
  差异比例直方图（6473 张图）。
- `figs/fig_g_margin.png` —— 分离边界图：各 run 最大 correct ratio 与最小有效负对照
  ratio 的水平线，标出 k=4/8/16/32 的位置，阴影区间是"在当前数据下可行的 k 范围"。

## 8. 与冻结协议的关系

本报告的全部"PASS/FAIL"都来自 `trace`/`isolated.calibrated`，是诊断证据，不是
`verify`。CamoFormer 冻结协议（`fp32-inference`，启用 source TF32）下的正式验收结果
**保持不变、保持失败**：`PARITY_FAILED`，详见
`migrations/CamoFormer/outputs/dsh-web-camoformer/README.md` 第 4 节。本报告中"正确基线通过
noise-calibrated"不能、也没有被用来声称冻结任务已经通过。

## 9. 威胁有效性

1. **k=8 是用同一批数据选出来再用同一批数据验证的，存在循环论证。** 本报告第 6 节
   给出的"可行 k 区间"(3.894, 25.974) 本身就是从这 13 组运行反推出来的；k=8 落在区间
   内这件事不能独立证明 k=8 对未见过的模型、未见过的突变类型同样有效。
2. **只有 3 个"有效"负对照（且全部是算子语义改动），样本量太小。** `l2_eps` 在当前
   输入上没有触发任何数值效果，不能算作已验证的负对照；还缺少形状错误、参数顺序
   错误、精度丢失（如误用低精度累加）、控制流分支错误等其他类别的负对照。见
   `experiment-plan.md` 的突变分类表。
3. **只在一个真实迁移任务（CamoFormer）、一个候选实现上标定。** 没有验证过这套规则
   在结构不同的模型（例如纯 CNN、纯 Transformer、RNN/状态空间模型）上是否同样能
   分离正确实现与错误实现。
4. **随机权重基线的"正确性"本身依赖于迁移实现与参数映射正确，不依赖任务级精度。**
   见第 5 节，模块隔离用的是随机权重或部分真实权重，不是完整训练 checkpoint；真实
   checkpoint 的任务级指标差异虽然很小，但这是**另一组实验**，不能倒推"随机权重下
   noise-calibrated 通过"等价于"任务精度也会通过"。
5. **只覆盖 2 种 GPU（sm_86 / sm_70）、1 个 Jittor 版本（1.3.9.14）。** 没有验证过更新或
   更旧的 Jittor 版本、AMD GPU、CPU-only 后端下 target 自身噪声的量级是否相近。
6. **target（Jittor）端存在尚未解释的运行间不确定性，而且本报告直接把这种不确定性
   当作"噪声的一部分"吸收进 k 的安全系数，而不是显式建模。** 固定输入、参数和代码后，
   target 重复执行的 mismatch 元素数仍在数千到数万量级（见图 e），`conv_opt=0` 更明显。
   如果某次具体运行恰好落在 ratio 接近 k 的边界，重复 `verify` 可能得到不同的
   PASS/FAIL。`compatibility/p3/jittor-reduce-nondeterminism.md` 记录了初步的根因线索
   （CUDA 并行归约的 atomicAdd 顺序），但尚未给出可验证的修复。
7. **`conv_opt=0` 证明 target 后端配置本身会显著改变 ratio（本报告测得 248.3，超过
   k=8），但只测试了一种变体。** 没有系统扫过 Jittor 的其他后端开关（tensor core、
   JIT kernel 搜索、cudnn 算法选择等）对 ratio 的影响，不知道 k=8 的安全余量在多少种
   后端配置下仍然成立。
8. **本报告的"模块位置/深度"用的是 trace 记录顺序（`position` 字段），是近似执行顺序，
   不是模型定义里的层深度**；图 (d) 的横轴解释需要带着这个限制看。

## 10. 推广到正式 verify 协议之前还需要什么

- **held-out 迁移验证**：至少在 BiRefNet、VSCode-v2、PlantCamo 三个未参与本次 k 标定的
  仓库上，用同一规则（k=8、全模块 gating）重复这组实验，使用由 DeepSeek 迁移（而不是
  本次产生 CamoFormer 候选的同一 agent/流程）产生的候选实现，检验 k=8 是否仍能分离
  正确与错误实现。这组仓库目前都还没有产出候选或 trace 证据。
- **更多突变类别**：至少补齐形状/维度错误、参数顺序错误、精度丢失、控制流分支错误、
  数值稳定性 trick 缺失（如遗漏 softmax 的 max-subtraction）等类别，覆盖
  `experiment-plan.md` 列出的约 12 类。
- **k 的灵敏度扫描**：在更大的正确/错误样本集上扫 k=2..32，画出检出率 vs 虚警率曲线，
  不能只报一个点值。
- **target 自身噪声的显式处理**：要么证明 target 的 run-to-run 噪声本身可以被纳入
  `source_alt` 式的噪声标定（即把 target 也跑两次，取其差作为 target 侧噪声的一部分），
  要么确认 k=8 的安全系数已经足够吸收当前观测到的 target 噪声量级（本报告图 e 显示
  target 自重复 max_abs 在 1e-3~1e-1 量级，取决于 run），要么说明为什么现状可以接受。
- **多输入**：本报告的全部模块隔离证据只使用一组固定输入；需要在多组不同输入下重复，
  确认 ratio 的量级不是某一组输入的偶然结果。
- **真实 checkpoint 下的模块隔离**：目前真实 checkpoint 只用于任务级指标，没有用于
  模块隔离 trace；需要补一组真实 checkpoint 下的 noise-calibrated trace，与随机权重结果
  对照。

在以上条目补齐之前，这套规则应继续作为 `trace`/`localize` 的诊断证据使用，不应写入
正式 `verify` 协议或替代冻结任务的数值验收结果。
