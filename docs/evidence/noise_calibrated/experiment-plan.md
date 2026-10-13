# Noise-calibrated 规则：协议升级实验计划

本文件是把 `docs/evidence/noise_calibrated/README.md` 中的"探索证据"推进到"可以写入
正式 `verify` 协议"所需要的具体实验设计。不在这里下结论，只定义要跑什么、怎么跑、
怎么判定成功/失败。

## 1. 突变目录（约 12 类，覆盖检测率/虚警率实验）

每类突变需要：注入方式、触发该突变所需的输入特征（有些突变只在特定输入形状/数值
范围下才会体现数值差异，例如本次 `l2_eps` 在当前随机输入下无效果）、预期受影响的
模块范围（叶子 or 组合）、以及是否已在 CamoFormer 上验证。

| # | 类别 | 注入方式 | 触发输入要求 | 预期受影响范围 | 已验证 |
| --- | --- | --- | --- | --- | --- |
| 1 | LayerNorm eps 不符 | 改 `eps` 常量（如 1e-5 → 1e-3） | 任意输入；eps 影响在输入方差小时更明显 | 叶子（norm 模块本身） | 是（`ln_eps`，CamoFormer） |
| 2 | 残差掩码作用范围错误 | 把 mask 从"只作用于 attention 内部 q/k"改成"作用于整个残差分支" | 需要 mask 非恒为 1（即输入里确实有被掩蔽的位置） | 组合模块（叶子不可见，见本报告图 c） | 是（`mask_residual`，CamoFormer） |
| 3 | GELU 近似公式错误 | 精确 erf 公式换成 tanh 近似 | 输入分布偏离 0 附近时近似误差更大 | 叶子（激活函数本身）及其下游 | 是（`gelu_tanh`，CamoFormer） |
| 4 | L2 归一化 eps 不符 | 改 `l2_normalize` 的 eps | 需要被归一化向量的范数接近 0（当前 CamoFormer 输入范数远大于 eps，故无效） | 叶子 | 已尝试但当前输入无效（`l2_eps`，CamoFormer），需要换输入或换到范数可能很小的位置重测 |
| 5 | 形状/维度顺序错误 | 转置或 reshape 时维度顺序写反（例如 NCHW vs NHWC 混用） | 任意输入；只要形状不是方阵/对称就能触发 | 该算子及全部下游（通常整网结构检查会先失败，不一定能走到数值阶段） | 未验证 |
| 6 | 参数加载顺序/切片错误 | state_dict 按名字正确但某个多头拆分的 reshape 顺序错误（如 QKV 合并权重拆分时 head 维度和 hidden 维度顺序反了） | 任意输入；多头数 > 1 时才能体现 | 该注意力模块及下游 | 未验证 |
| 7 | 精度丢失（低精度累加） | 把某个 reduce（sum/mean）的累加器类型从 fp32 降到 fp16 | 需要较长的累加序列（如大 kernel 的卷积、大 seq_len 的 attention）才会体现误差 | 叶子（该 reduce 算子） | 未验证 |
| 8 | 数值稳定性 trick 缺失 | softmax 漏掉 max-subtraction，或 LayerNorm 漏掉方差下限保护 | 需要输入数值范围较大（如未归一化的 logits）才会体现（输入值很小时等价） | 叶子，但误差可能在极端输入下非常大 | 未验证 |
| 9 | 控制流分支错误 | 训练/推理分支弄混（如 Dropout 在 eval 模式下没有正确关闭，或 BatchNorm 用了 batch 统计量而非 running 统计量） | 需要 batch size > 1 或多次前向才能体现（单样本 eval 有时会被 BatchNorm 的 running stats 掩盖） | 该模块及全部下游 | 未验证 |
| 10 | 插值/下采样对齐方式错误 | `align_corners` 或 `mode` 不一致（如 bilinear vs nearest，或 align_corners=True vs False） | 需要空间分辨率变化的下采样/上采样操作 | 该算子及下游 | 未验证 |
| 11 | 权重初始化依赖的隐式行为差异 | 这不是"迁移错误"而是"框架默认行为差异"类，例如 Conv 的 padding 模式默认值不同（zeros vs reflect） | 任意输入；取决于是否接近边界 | 该算子及下游，通常在图像边缘像素更明显 | 未验证；需要先确认这是否算"迁移问题"还是"框架语义差异"，分类前先查 PyTorch/Jittor 文档 |
| 12 | 算子语义近似但非等价（如不同的 attention 实现路径，flash-attention vs 显式 QK^T softmax V） | 两种数学上应该等价但数值路径不同的实现 | 任意输入；差异量级通常接近"正常后端噪声"而非"明显错误"，是本规则最难分离的一类 | 组合模块及下游 | 未验证；预期这类突变的 ratio 可能落在"可行 k 区间"内部，是检验 k=8 是否"太松"的关键类别 |

对每一类，具体实验步骤：

1. 在隔离的候选工作区分支上实现该突变（不改 `src/`、不改冻结任务定义）。
2. 用与本次相同的 `trace` 流程（`isolated.calibrated`，k=8，全模块 gating）跑一次。
3. 记录：是否被规则拒绝（ratio > k）、首个失败模块、ratio 量级、是否需要特定输入才能
   触发（如类别 4、7、8、10 需要专门构造触发输入，不能直接复用当前随机输入）。
4. 汇总成"检出率"（12 类里有多少类在至少一种输入下被规则正确拒绝）和"虚警率"
   （4 个正确基线配置里有多少被误判为失败——目前是 0/4，需要在扩充后的正确配置集合
   上重新统计）。

## 2. Held-out 仓库

目标：验证 k=8 不是只对 CamoFormer 这一个模型、这一个候选实现有效。仓库选型已经完成一轮
公开信息调研，见 `docs/audits/heldout-cod-survey.md`（2026-10-08，只读 GitHub 公开信息，
未下载权重/数据集，未实测显存与 trace 体量）；以下状态直接取自该文档，不是本次新调研。

| 仓库 | 冻结 commit | 输入分辨率 | 权重获取方式 | 现有四套 COD 数据适配 | 迁移执行者 | 当前状态 |
| --- | --- | ---: | --- | --- | --- | --- |
| `ZhengPeng7/BiRefNet`（BiRefNet-COD） | `ebcc0bc8ec7f...`（main，2026-09-02）；tag `v1` | 1024 | GitHub release 直链 `BiRefNet-COD-epoch_125.pth`，885,082,437 字节，可非交互下载 | 四套齐全，但目录名需适配（`TE-CAMO→CAMO`、`TE-COD10K→COD10K`） | DeepSeek（模型 API，非本次产生 CamoFormer 候选的流程） | 未开始实验；仓库与权重均可直接取得，调研判定为可行性最高的主候选，但 trace 风险"很高"（未实测）：1024²、Swin-L、deformable ASPP 会产生大张量，首轮需限制保存粒度 |
| `Sssssuperior/VSCode`（VSCode-v2-T stage2） | `d0ec32e2f1b2...`（main，2025-12-08）；无 tag | 352 | 仅 Baidu 网盘提取码页面（`8imx`），无 GitHub release/HF 直链，文件大小未公布 | 只有 COD10K/CAMO/NC4K，缺 CHAMELEON；目录名也需适配 | DeepSeek | 未开始；**权重获取是当前阻塞**（需要人工从 Baidu 下载后以隐藏、可校验的本地 artifact 形式提供，不能自动化非交互获取） |
| `yjybuaa/PlantCamo` | `867452b7b43c...`（main，2024-12-11）；无 tag | 704 | Google Drive / Baidu `view`/提取码页面，无直链 | 官方只测自建 PlantCAMO1250（`rgb/gt`），**不适配**现有 COD10K/CAMO/NC4K/CHAMELEON | DeepSeek | 未开始；权重获取同样受阻；且官方测试集与现有四套 COD 数据不是同一任务，若纳入需要在分析里单独说明，不能与另两个仓库的成功率直接横比 |

调研还列出了 `ChunmingHe/RUN`（原生匹配四套数据目录，但依赖 `mamba-ssm`/Triton 自定义
CUDA selective scan，构建兼容性是先决阻塞，trace 风险"很高"）和 `ai4colonoscopy/PraNet-V2`
（息肉分割任务，官方已有 Jittor 推理，作为"非迁移对照"而非验证集使用）。本计划的三个主
held-out 候选只取 BiRefNet、VSCode-v2、PlantCamo，按任务输入要求保持一致；RUN 留作后续
"高难候选"，不纳入本轮 k 验证时间表。

权重获取阻塞是实验计划里需要先解决的前置条件，不是实验设计本身的缺陷：BiRefNet 可以立即
开始，VSCode-v2 和 PlantCamo 需要先有人工下载并校验哈希的本地权重副本（遵守"密钥、下载
凭据不进仓库"的约束，只是普通模型权重文件，不涉及红线）。

每个仓库的实验步骤与 CamoFormer 一致：

1. 用 DeepSeek 产生一次候选迁移（走正常迁移流程，不是本次的人工构造突变）。
2. 对候选运行 `trace`，拿到真实的 `isolated.calibrated` 证据（这是"正确性未知"的真实
   候选，不是预先知道答案的负对照）。
3. 独立判断候选是否语义正确（人工审查代码 diff + 跑 `verify` 看是否通过冻结协议），
   作为 ground truth。
4. 比较 k=8 的判定与 ground truth 是否一致：
   - 若候选语义正确但 `verify` 因为后端数值路径失败（预期中的常见情况，类似
     CamoFormer），检验 noise-calibrated 的 ratio 是否仍然 ≤ k=8。
   - 若候选语义有错误，检验 ratio 是否 > k=8，且首个失败模块是否指向真实错误位置。
5. 不能用这三个仓库的结果去重新调整 k——它们是验证集，不是训练集；如果 k=8 在这里
   失效，应该如实报告"k=8 不能推广"，而不是改 k 之后用同一批数据再证明一次。

## 3. k 灵敏度扫描（2..32）

在现有 CamoFormer 证据（13 组运行）基础上，额外需要：

1. 扩充正确配置数量（目前只有 3 个：3080 Ti 随机权重、3080 Ti 真实 encoder、V100
   随机权重）——至少补充"真实 decoder 权重，encoder 随机"这种互补配置，以及更多随机
   种子下的随机权重基线（当前每个 GPU 只有 1 个随机种子）。
2. 补齐第 1 节的其余约 9 类突变。
3. 对每个 k ∈ {2, 3, 4, 6, 8, 11, 16, 22, 32}（几何间隔采样），计算：
   - 检出率 = 被正确拒绝的"确认有效"突变数 / 全部"确认有效"突变数
   - 虚警率 = 被误判为失败的正确配置数 / 全部正确配置数
   - 画 ROC 风格曲线（检出率 vs 虚警率，k 作为参数遍历曲线上的点），而不是只报告
     单个 k 下的两个数字。
4. 如果存在候选 k 使虚警率为 0 且检出率为 1（即完全分离），报告该 k 的区间，而不是
   单点；如果不存在完全分离的 k（预期类别 12"语义近似但非等价"的突变最可能导致
   这种情况），需要明确报告"当前 target 噪声水平下，k 无法同时做到零虚警与完全检出"，
   并讨论是否需要更精细的噪声标定（如按模块类型分别标定 k，而不是全局单一 k）。

## 4. Target 自身噪声的处理

当前规则只用 source 的两条路径（fp32 vs fp64、fp32 vs alt）估计噪声，没有显式使用
target 的 run-to-run 噪声。需要补充的实验：

1. 对每个已有的正确配置，把 target 侧也重复执行 N 次（N ≥ 5），记录每次与 fp64
   source 的 ratio，观察 ratio 本身的 run-to-run 波动范围（而不是只看一次 ratio）。
2. 检验"k=8 的安全余量"是否已经覆盖了这个波动范围：若某次 target 执行恰好把 ratio
   推到接近 8 的边界，重复 N 次里是否会出现跨越边界的情况。
3. 如果波动范围显著（根据本次图 e，target 自重复的 `max_abs` 在 1e-3~1e-1 量级，
   取决于配置，`conv_opt=0` 明显更大），评估两种修复方向，不预设哪个更好：
   - 方向 A：把 target 的 run-to-run 差异也计入"分母"，即 `noise = max(source_fp64_err,
     source_alt_err, target_repeat_err)`，用更保守的噪声估计。
   - 方向 B：要求 `verify` 对 target 侧也做多次重复取多数判定，而不是依赖单次执行。
     这与 `AGENTS.md`"不能挑选最好的一次、平均多次输出"的约束需要协调：多次重复
     判定一致性（如 5 次里 5 次都 PASS）与"取最优结果"是不同的事情，需要在正式协议
     文件里明确写清楚允许哪种多次执行语义。

## 5. Jittor 不确定性证据补充（独立于 noise-calibrated 规则本身）

目标：把 `compatibility/p3/jittor-reduce-nondeterminism.md` 当前"observed，未确认修复"
的状态往前推进，产出可以回答"哪些配置下 target 是确定性的、哪些不是"的矩阵证据。

矩阵设计：

| 维度 | 取值 |
| --- | --- |
| Jittor 版本 | 1.3.9.14（当前已验证）；需要补充至少一个更新版本做对照 |
| GPU / CUDA | sm_86/CUDA 12.8（3080 Ti，已验证）、sm_70/CUDA 11.3（V100，已验证）；可选补充其他架构 |
| `conv_opt` | 0、1（当前 `jittor-reduce-nondeterminism.md` 已覆盖 1；0 在 CamoFormer 整网上"更差"但未在算子级隔离重测） |
| 粒度 | 单算子（复用 `probe_jt_determinism.py` 的模式：conv1x1、conv3x3、depthwise、linear、sum(dim=1)、mean(dim=-1)、interpolate）；整网（复用 `probe_jt_wholenet.py`） |
| 重复次数 | 每个组合至少 5 次（与现有 `probe_jt_determinism.py` 一致） |

执行步骤：

1. 复用 `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/probe_jt_determinism.py` 作为单算子
   探针的基础，扩展其 `for use_tc in (0, 1)` 循环，加入 `conv_opt` 维度（需要在子进程
   启动前设置环境变量，而不是进程内切换，因为 `conv_opt` 影响 Jittor 初始化时的 cuDNN
   加载，参考 `compatibility/p3/jittor-reduce-nondeterminism.md` 第 74-88 行的源码证据）。
2. 复用 `probe_jt_wholenet.py` 做整网粒度的重复执行，记录每次输出与第一次的最大绝对差、
   mismatch 元素数（与本次 `isolated.repeatability` 的统计口径一致，便于与现有
   CamoFormer 证据直接比较）。
3. 复用 `probe_jt_firstnd.py` 定位每个配置下"第一个出现不确定性的算子"，与
   `jittor-reduce-nondeterminism.md` 已记录的 `atomicAdd` 机制交叉验证：检查该算子的
   生成 CUDA 代码（`*_op.cc`）是否确实包含 `atomicAdd`，是否是 `ReduceOp`/融合 Conv2d
   图。
4. 产出结果矩阵：每个 (Jittor 版本 × GPU × conv_opt × 粒度) 组合的"5 次重复的最大
   run-to-run 差异"，标注哪些组合是逐位稳定的（如本次已知的 depthwise Conv2d、Linear、
   `mean(dim=-1)`、`interpolate`），哪些不稳定（如 `conv_opt=1` 的普通 Conv2d、
   `sum(dim=1)`）。
5. 把该矩阵补充进 `jittor-reduce-nondeterminism.md` 的 `applicability.known` 字段，
   不新开一个问题条目（遵守 P3 问题库"不建立第二套根因系统"的约束）。

本节不要求在当前任务里产出新的根因修复；目标只是把"已知不确定、已知确定"的边界
从当前的"两种 GPU、一种 conv_opt 下零散验证"扩展成系统矩阵，供后续决定是否需要
把 target 噪声显式纳入 noise-calibrated 规则（见第 4 节）。
