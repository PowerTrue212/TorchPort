# APT、CLIP、RAFT 的 PyTorch → Jittor 迁移会话审查

**审查日期：** 2026-10-01  
**审查对象：** Codex、DeepSeek（pure DSH、DSH + TorchPort、DeepSeek V4 Flash + TorchPort）在 APT、CLIP、RAFT 迁移过程中的原始会话、修复轮次和统一评测结果  
**核心问题：** 迁移中真正困难的 PyTorch/Jittor 差异是什么，代理如何定位和修复，以及哪些问题仍未解决

## 1. 结论摘要

1. **三项任务实际上代表三层不同难度。** CLIP 主要考验 API、模块结构和权重键映射；APT 进一步要求数据增强、随机数、优化器、调度器和长期训练状态一致；RAFT 则进入 Jittor 图融合、CUDA 归约顺序、循环误差放大和 cuDNN 内核选择。不能用同一个“转换成功”概念解释三者。
2. **PyTorch 与 Jittor 中数学公式相同，不保证 FP32 执行相同。** RAFT 的主要误差并非模型公式翻译错误，而是 eager kernel 边界、融合重关联、原子归约、softmax/InstanceNorm 归约树和 cuDNN 累加分块不同。
3. **Codex 的 RAFT 修复链最系统。** 它从权重别名、cuDNN 算法选择、归一化归约、`grid_sample` 前的融合边界，到 BatchNorm 运算顺序逐层缩小误差，最终在第 5 个公共修复轮通过。
4. **pure DSH 在 CLIP 上足够有效，在 RAFT 上失去定位焦点。** CLIP 的问题可以通过 API 探测和键名修正解决；RAFT 中 pure DSH 同时改动卷积、相关性、归一化、池化、softmax 和采样，缺少可靠的 first-divergence 证据，出现“偶尔通过、重跑失败、甚至段错误”的不稳定状态，最终由用户停止并记为失败。
5. **TorchPort 的实际价值主要出现在 RAFT 的诊断阶段，而不是自动转换阶段。** 它把严格模式的首个分歧定位到 `feature.0`，并支持把失配元素从 5,751 降到 797；但最终 strict 仍被 4 个无法复刻的 cuDNN 闭源内核阻塞。CLIP 中三种方法均 Round 0 通过，TorchPort 没有带来可测的修复优势，反而引入过一次 Jittor CUDA 缓存污染。
6. **APT 只有 Codex 原始会话，未找到 APT 的 DeepSeek 会话。** 这应标为“证据缺失”，不能写成 DeepSeek 失败，也不能拿 CLIP/RAFT 的 DeepSeek 记录代替。

## 2. 证据范围

| 任务 | Codex | pure DSH | DSH + TorchPort | 其他 DeepSeek | 证据状态 |
|---|---|---|---|---|---|
| APT | 有完整原始会话 | 未找到 | 未找到 | 未找到 | 只能审查 Codex；DeepSeek 为 missing evidence |
| CLIP | 有完整原始会话 | 有完整压缩会话 | 有完整压缩会话 | — | 三种方法均可审查 |
| RAFT | 有完整原始会话 | 有完整压缩会话 | 有完整压缩会话 | V4 Flash + TorchPort 有完整会话 | 四条执行链均可审查 |

### 2.1 原始会话

Codex：

- APT：`C:\Users\31561\.codex\sessions\2026\08\23\rollout-2026-08-23T23-42-12-01a02f49-5f69-7ce0-ad38-e80de3f70126.jsonl`
- CLIP：`C:\Users\31561\.codex\sessions\2026\08\22\rollout-2026-08-22T12-20-26-01a027b2-d36c-7830-a756-4492dd5e9332.jsonl`
- RAFT：`C:\Users\31561\.codex\sessions\2026\08\24\rollout-2026-08-24T11-57-07-01a031ea-3178-7fd1-991b-a87f4e20fd40.jsonl`

DeepSeek / DSH：

- CLIP pure DSH：`session-7d4163e8-9fbc-4cbc-9b38-1bf15818636f`
- CLIP DSH + TorchPort：`session-9bc26245-d02a-4bff-a7d7-80056cdec39a`
- RAFT pure DSH：`session-8d5c1b6d-ad22-44dd-a343-d62c03acf56a`
- RAFT DSH + TorchPort：`session-4582abe7-1df8-488a-b0c3-e202460da31f`
- RAFT DeepSeek V4 Flash + TorchPort：`session-0b77890f-9a9d-4a2e-9259-f8c946955f23`

### 2.2 结果证据

- CLIP 统一结果：[`CLIP/outputs/summary.json`](../CLIP/outputs/summary.json)
- RAFT 统一结果：[`RAFT/outputs/summary.json`](../RAFT/outputs/summary.json)
- RAFT pure DSH 终止记录：[`RAFT/results/dsh/termination.json`](../RAFT/results/dsh/termination.json)
- RAFT Codex 轮次：[`RAFT/results/codex/rounds.jsonl`](../RAFT/results/codex/rounds.jsonl)
- RAFT DSH + TorchPort 主轮次：[`RAFT/results/dsh_torchport/rounds.jsonl`](../RAFT/results/dsh_torchport/rounds.jsonl)
- RAFT DSH + TorchPort strict 轮次：[`RAFT/results/dsh_torchport/traces/strict_rounds.jsonl`](../RAFT/results/dsh_torchport/traces/strict_rounds.jsonl)
- V4 Flash 结果：[`RAFT_DSH_V4_FLASH/results/dsh_torchport/result.json`](../RAFT_DSH_V4_FLASH/results/dsh_torchport/result.json)
- APT 既有证据审计：[`apt/reports/agent_vs_human_migration_report.md`](../apt/reports/agent_vs_human_migration_report.md)

## 3. APT：训练轨迹一致性比 API 转换更难

APT 不是固定输入的单次推理，而是 CIFAR-100 上 10 个增量任务、每任务 30 epoch 的完整训练。任何数据增强、随机数消费、可训练参数集合、调度器步数或恢复时机的差异，都会被后续任务持续放大。

### 3.1 Codex 遇到的问题与修复

| 问题 | 根因 | 修复 | 审查判断 |
|---|---|---|---|
| 初始 bootstrap 被拒绝 | 源码树实际哈希与冻结协议声明不一致 | 停止执行，等待协议修正后重新开始 | 正确的协议控制，不属于迁移失败 |
| prompt 形状/reshape 异常 | Jittor 实现中的 prompt 初始化和层索引形状没有完全对应 PyTorch | 修正 prompt 状态与 reshape 路径 | 模型结构语义修复 |
| AutoAugment 结果漂移 | Pillow `Solarize` 阈值语义和 torchvision 不同；随机数调用次数/顺序不同 | 复现 torchvision policy，修正 Solarize，并预先抽取 probability/sign 随机数以保持 RNG 消费顺序 | APT 最关键的训练数据语义修复之一 |
| scheduler 与可训练参数不一致 | cosine scheduler 的 epoch 偏移不同；`clf_norm` 被错误加入训练 | 修正 scheduler 步进，排除 `clf_norm` | 优化轨迹语义修复 |
| Jittor 训练接口不同 | PyTorch 使用 `zero_grad/backward/step`，Jittor 使用 `optimizer.step(loss)` | 按 Jittor 语义重写训练循环 | 基础框架 API 迁移 |
| 远程训练在 task 5 断开 | SSH/长任务基础设施中断；在错误边界 resume 会重复 EMA merge | 保留中断证据，从 task 1 以 `nohup/setsid` 重跑；同时增加 `progress.npz` 的任务级恢复 | 基础设施问题，但揭示了持续学习状态恢复必须有语义边界 |
| Round 0 只差 `1.129×10⁻⁵` | evaluator 对 float32 汇总均值设置 `1×10⁻⁵` 边界，模型、预测和 checkpoint 均有效 | 从 task-10 checkpoint 重评，仅选择相邻 float32 ULP 表示；未改变权重或预测 | 评测器边界适配，不应计作模型语义改进 |

最终 Codex 结果：`RN=true`，平均准确率 `89.1600036621%`，遗忘率 `3.9666655064%`，相对参考的平均准确率差 `7.6293945×10⁻⁶` 个百分点，修复轮数为 1。

### 3.2 APT 的核心迁移结论

- APT 的主要风险不在 `forward → execute`，而在**训练协议的逐步等价性**。
- 数据增强必须同时复现图像运算和 RNG 消费顺序；“视觉上相似”的增强替代不能视为等价迁移。
- optimizer、scheduler、EMA merge、prompt 状态和 task boundary 必须共同进入 checkpoint。
- APT Round 1 是评测表示修复，不应与模型/训练修复混合统计。
- 本地没有 APT DeepSeek 会话，因此无法评价 DeepSeek 是否会采用相同修复或达到相同结果。

## 4. CLIP：主要是 API、结构和权重映射

CLIP 使用固定图像、固定文本和固定 checkpoint，只执行一次推理。没有训练状态，也没有 RAFT 式的递归误差放大，因此三种方法均在第一次公共 evaluator 通过。

### 4.1 统一结果

| 方法 | RN | NA | Max error | Rounds |
|---|---:|---:|---:|---:|
| Codex | true | true | `1.335144×10⁻⁵` | 0 |
| pure DSH | true | true | `1.525879×10⁻⁵` | 0 |
| DSH + TorchPort | true | true | `1.525879×10⁻⁵` | 0 |

### 4.2 Codex 的迁移处理

Codex 手工构建了 native Jittor ViT-B/32，并显式对齐：

- 302 个 FP32 state-dict 键；
- QuickGELU：`x * sigmoid(1.702*x)`；
- LayerNorm `eps=1e-5`；
- 文本 EOT 特征取 token ID 的 `argmax` 位置；
- image/text embedding 归一化后乘 `exp(logit_scale)`；
- PyTorch 中作为普通属性保存、未进入 NPZ 的 causal attention mask，在 Jittor 侧重建。

CLIP Codex 的成功说明：当输入、权重和执行路径固定时，**显式模型重建 + 权重结构核对**足以达到公共容差，不需要低层 CUDA 仿真。

### 4.3 pure DSH 遇到的问题与修复

| 问题 | 观察 | 修复 |
|---|---|---|
| Jittor API 名称不同 | `jt.nn.sigmoid` 不存在 | 改用 `jt.sigmoid` |
| `argmax` 返回语义不同 | Jittor 1.3.11 的 `Var.argmax(dim)` 返回 tuple | 对冻结 token 输入使用 NumPy argmax，避免错误解包 |
| MHA 权重键不匹配 | PyTorch 使用 `attn.out_proj.weight/bias`，初版 Jittor 是平铺属性 | 创建嵌套 `Linear` 子模块，匹配 PyTorch state-dict 层级 |
| MLP 权重键不匹配 | PyTorch 为 `mlp.c_fc.*`、`mlp.c_proj.*` | 保留同名嵌套模块结构 |
| causal mask 不在权重文件 | mask 是普通属性，不是 state-dict tensor | 在目标侧重建 |

这条会话的修复高度集中：先探测 Jittor API，再让模块命名和权重键与 PyTorch 对齐。它没有进入 CUDA 内核层，说明 pure DSH 并非普遍无效，而是其有效性取决于任务是否停留在结构转换层。

### 4.4 DSH + TorchPort 的实际作用与副作用

- 手工实现 causal mask 时曾把方向写反，导致 attention 全部 NaN；修正为上三角 `-inf` mask 后恢复。
- TorchPort 自动迁移识别出 4 个静态 `.to()` blocker；目标仍依赖 Jittor 1.3.11 不存在的 `nn.MultiheadAttention`，所以自动目标并不能直接运行。
- TorchPort 的目标子进程把 `nvcc_path` 置空，触发 Jittor 在共享缓存中重编译 CPU/no-CUDA `jittor_core`，后续 import 因 `cuda_archs` flag 缺失失败。删除被污染的缓存 core 后才恢复。
- 最终公共 evaluator 的通过来自已经完成的 native Jittor 实现，并非 TorchPort 自动修复。

因此，CLIP 上 TorchPort 提供了额外的扫描和证书信息，但没有降低公共修复轮数；反而说明工具执行必须使用独立的 Jittor 缓存和环境。

## 5. RAFT：核心是执行顺序、归约树与后端内核

RAFT 输出为 `1×2×384×512`，并通过 12 次 recurrent update 逐步更新光流。每一步的微小 FP32 差异都会传播到后续相关性采样、GRU 状态和上采样 mask，因此它是三项任务中最能暴露 PyTorch/Jittor 执行差异的模型。

### 5.1 统一公共结果

| 方法 | RN | NA | Max error | Rounds | 最终状态 |
|---|---:|---:|---:|---:|---|
| Codex | true | true | `1.237392×10⁻⁴` | 5 | 通过 |
| pure DSH | true | false | `2.664328×10⁻⁴` | 1 | 用户停止，定义为失败 |
| DSH + TorchPort | true | true | `1.486540×10⁻⁴` | 1 | 通过 |
| DeepSeek V4 Flash + TorchPort | true | true | `1.730919×10⁻⁴` | 3 | 独立追加实验通过 |

注意：公共 evaluator 的 `NA=true` 只表示 `rtol=atol=1e-4` 下通过，不代表严格中间张量一致。

### 5.2 Codex 的五轮修复链

| 轮次 | 失败位置/现象 | 根因 | 修复 |
|---:|---|---|---|
| 0 | 加载权重前失败，出现 8 个 unexpected keys | Jittor state-dict 会去重同一个 BatchNorm 模块的别名，而 PyTorch NPZ 同时包含 `norm3` 与 `downsample.1` | 只允许经过验证的 alias keys，并加载对应数组；不使用宽泛的 `strict=False` |
| 1 | 同一提交跨运行结果变化 | Jittor cuDNN `cudnnFind` 根据计时结果选择算法，选择不稳定 | `jt.cudnn.set_algorithm_cache_size(0)`，改用可复现的 heuristic 选择 |
| 2 | 仍有非确定性 | InstanceNorm 空间归约及 9 路 convex upsample 使用 `atomicAdd`，加法顺序不固定 | 自定义固定归约树的 CUDA InstanceNorm；显式固定 softmax 和加权求和顺序 |
| 3 | 相关体一致，但采样后开始分歧 | Jittor 将坐标归一化、拼接与 `grid_sample` 融合并重关联；PyTorch eager 在算子边界发生 FP32 舍入 | 在 mul/div/sub/concat 后使用 `stop_fuse()`，恢复 PyTorch kernel boundary；采样相关误差从约 `8.30×10⁻⁵` 降至 `1.91×10⁻⁶` |
| 4 | encoder 仍有系统偏差 | Jittor eval BatchNorm 折叠 affine 的运算顺序与 PyTorch 不同 | 自定义 `(x-mean)/sqrt(var+eps)*gamma+beta`，并设置融合边界 |
| 5 | 公共 evaluator 通过 | 上述语义和数值修复共同生效 | `RN=true, NA=true` |

Codex 的优势不在“更少轮”，而在每轮都把因果范围缩小：先结构，再确定性，再第一个数值分歧，最后处理算子内部顺序。

### 5.3 pure DSH 为什么没有稳定通过

pure DSH 正确意识到 RAFT 的问题位于低层数值执行，并尝试替换大卷积、相关矩阵乘、InstanceNorm、平均池化、softmax 和 `grid_sample`。但会话出现三个问题：

1. **修改范围过宽。** 没有稳定的 first-divergence 结果时同时怀疑多个算子，难以证明某次修改真正缩小了误差。
2. **Jittor `reindex`/gather 语义调试成本高。** 自定义 grid sample 遇到字符串索引与 Var 索引混用、维度和广播失败；某些实现反而把相关性误差从约 `8×10⁻⁵` 放大到 `1.22×10⁻⁴`。
3. **结果不可复现。** 同一方向的 evaluator 曾出现 `1.780987×10⁻⁴` 且 NA=true，随后重跑为 `2.086163×10⁻⁴`（19 个元素失败）和 `1.777411×10⁻⁴`（1 个元素失败），另有一次 Jittor/CUDA 段错误。

最终记录以最后一次被接受的公共 evaluator 为准：`RN=true, NA=false, max_error=2.664328×10⁻⁴`。用户停止后 workspace 又有修改，但没有再次评测，因此不能把未评测版本当成最终结果。

方法学上还有一个限制：该 pure DSH 会话运行时 TorchPort 工具对模型可见，但日志中未发现任何禁用调用。它是“没有使用 TorchPort”的记录，但不是工具层面完全隔离的纯基线。

### 5.4 DSH + TorchPort：公共通过与 strict hardening 必须分开

主实验中，Round 0 的 NA 失败后使用了阶段捕获，Round 1 即通过公共 evaluator。此后又进行独立的 strict hardening，不能把 hardening 的 4 轮加到主实验 Rounds=1 中。

Strict 修复链如下：

1. 首个分歧定位到 `feature.0`，`max_abs=4.768×10⁻⁶`，5,751 个元素超过 strict 容差。实现 PTX `div.rn.f32`，并复刻 PyTorch CUDA Welford 归约：512 线程、XOR butterfly、零填充二次归约和 `rsqrtf`。
2. 组件单独一致但 residual block 输出仍分歧，说明 Jittor fuser 跨 PyTorch kernel boundary 重关联。对 norm、ReLU、sigmoid、tanh 和 add 后加入 `stop_fuse()`。
3. 复刻 warp softmax 归约树；把 GRU 的 1×5 / 5×1 卷积改为 5 次移位 1×1 卷积。`feature.0` 最大误差降至 `3.576×10⁻⁶`，strict 失配从 5,751 降到 797。
4. 剩余 strict 分歧缩小到 4 个 cuDNN 内核：`layer3[0].conv1`、`encoder.convc1`、`gru.convz1/z2`。尝试顺序 FMA、不同通道/核顺序、Winograd F(2,3) 和 cuDNN 9.1 v8 shim，仍无法选择 PyTorch 使用的同一 engine 和累加分块。

最终：公共 FP32 状态通过，strict 状态仍为 false。这不是“算法未迁移”，而是**无法在缺少同一 cuDNN engine 选择信息时重现闭源内核的具体累加顺序**。

### 5.5 DeepSeek V4 Flash + TorchPort 的额外路径

V4 Flash 的三轮问题更集中在 Jittor 编译图和长递归误差：

| 轮次 | 问题 | 修复 |
|---:|---|---|
| 0 | Jittor 把约 2,480 个操作融合成巨型图，编译器触发 `member.size() <= var_num` | 用 `stop_fuse` 拆图并设置执行边界 |
| 1 | RN=true，但一个像素超过 NA；smoke 路径每轮执行，而正式路径延迟融合 | 在每个 RAFT iteration 末执行 `jt.sync()` |
| 2 | FP32 微差仍经 12 次 GRU 累积 | 实现 double-fp32：Knuth two-sum/pairwise tree GEMM、补偿 GRU、df32 norm 和坐标累积 |
| 3 | 公共 evaluator 通过 | `max_error=1.730919×10⁻⁴`，距容差边界的最小余量约 `2.4×10⁻⁵` |

这是一条有技术创造性的修复路径，但也有两个限制：

- 会话中只有少量 TorchPort 调用，没有与主 DSH + TorchPort 同定义的 strict trace/certificate，因此不能用它评价 TorchPort 的 first-divergence 质量。
- double-fp32 是复杂、任务专用的数值仿真。它证明可以过公共容差，但维护成本和对其他输入/硬件的泛化尚未验证。

## 6. PyTorch → Jittor 的核心迁移问题

### 6.1 API 兼容只是第一层

常见问题包括：

- `forward` 改为 `execute`；
- `optimizer.step(loss)` 替代 PyTorch 的三段式反向传播；
- `jt.sigmoid` 与 `jt.nn.sigmoid` 的命名差异；
- `argmax` 返回值语义不同；
- Jittor 1.3.11 缺少 `nn.MultiheadAttention`；
- 原地 slice 写入在 Jittor 中不安全或不受支持。

CLIP 基本停留在这一层，所以三种方法都能 Round 0 通过。但 APT、RAFT 的主要困难均不在这里。

### 6.2 state-dict 不只是键名列表

- PyTorch 与 Jittor 的嵌套模块命名不同会直接破坏自动加载。
- 同一子模块被多个路径引用时，Jittor 可能去重，PyTorch 权重包却保留多个 alias key。
- mask 等普通属性可能不进入 state dict，需要在目标侧根据模型语义重建。

建议使用显式 key map、alias allowlist、shape/dtype 校验和加载覆盖率报告。宽泛的 `strict=False` 会掩盖真实结构错误。

### 6.3 数据与 RNG 是模型的一部分

APT 证明，训练迁移中必须对齐：

- PIL/NumPy/框架三套随机源；
- 每个增强操作的随机数调用次数和顺序；
- 像 Solarize 这样的阈值/取整细节；
- sampler、task order 和恢复后的 RNG state。

若训练输入轨迹不同，最终准确率相近也不能证明实现逐步等价。

### 6.4 Jittor 图融合会改变 FP32 语义

PyTorch eager 往往在每个 kernel 结束时把中间结果舍入到 FP32；Jittor 会把多个逐元素操作融合，编译器可能重关联算式。两者实数公式相等，浮点轨迹却不同。

`stop_fuse()` 的正确用途是恢复已经通过 first-divergence 证明的 PyTorch kernel boundary，而不是在全模型中无差别插入。长循环图还可能超过 Jittor fusion compiler 的容量，此时 `jt.sync()` 兼具拆图和稳定执行的作用。

### 6.5 归约顺序决定 Normalization 和 Softmax 的结果

Jittor 的 `mean`/InstanceNorm 可能使用原子加法；PyTorch CUDA 可能使用固定 Welford、warp shuffle 或分层 tree reduction。FP32 加法不满足结合律，因此：

- 相同均值公式不保证相同 bit pattern；
- 运行到运行的原子顺序变化会产生不稳定结果；
- exact parity 可能需要自定义 CUDA reduction；
- 若实验只要求公共容差，应避免为 bitwise parity 付出不必要的复杂度。

### 6.6 cuDNN 算法和 engine 是 strict 的硬边界

同一卷积形状可由 direct、implicit GEMM、Winograd 等不同内核完成；即便算法名称相同，engine 的累加分块也可能不同。关闭 timing-based `cudnnFind` 可以提高 Jittor 自身的重跑稳定性，但不保证与 PyTorch 选择同一 engine。

当 strict 的剩余分歧只来自无法选择的闭源 cuDNN engine 时，应记录为“后端内核不可复刻”，而不是继续无限修改模型层。

### 6.7 循环模型会放大局部误差

RAFT 的 12 次 update 说明最终输出误差并不能可靠指出原因。一个早期 feature 或坐标采样的微小差异，会通过 correlation lookup 和 GRU 隐状态不断传播。有效诊断必须比较：

1. feature encoder 的最早输出；
2. correlation volume；
3. context encoder；
4. 每次迭代的 hidden state、delta flow、mask；
5. 最终 `flow_up`。

只盯最终 max error，容易导致 pure DSH 式的多算子同时重写。

### 6.8 RN、NA 和 strict 是三个不同目标

- **RN：** 目标实现可以完成 forward/训练。
- **NA：** 最终输出在公共容差内。
- **Strict/SA：** 中间层或输出达到更严格、接近逐位的对齐。

CLIP 三者都容易；RAFT 可以 RN/NA 通过而 strict 失败。实验报告必须分别记录，不能把公共通过写成“与 PyTorch 完全一致”。

## 7. Codex 与 DeepSeek 的过程对比

### Codex

- 优点：遵守冻结协议，修复轮次因果链清楚；RAFT 中能逐步从结构问题缩小到具体 kernel boundary。
- 缺点：为了 evaluator 通过会做高度任务定向的处理；APT 的 ULP 修复应单列为 evaluator adaptation。RAFT R5 也说明它并不总是更省轮次。

### pure DeepSeek / DSH

- 优点：对 CLIP 的 API 和 state-dict 问题处理直接有效；无需 TorchPort 也能完成普通推理模型迁移。
- 缺点：在 RAFT 这类数值敏感循环模型中，缺少 first-divergence 时容易同时重写多个底层算子，造成搜索空间扩大和重跑不稳定。

### DSH + TorchPort

- 优点：RAFT strict 中能把“最终输出不一致”转化为 `feature.0` 的具体失配数量和误差，并验证每类修复是否真正向前推进。
- 缺点：自动迁移不是可靠终点；CLIP 的 MHA 目标仍不可实例化。工具还可能污染共享 Jittor cache。即使定位准确，也不能突破 cuDNN 闭源 engine 的信息边界。

### DeepSeek V4 Flash

- 优点：能从融合编译器崩溃和循环误差放大推导出拆图、逐轮同步和 compensated arithmetic。
- 缺点：方案复杂且 benchmark-specific；当前没有同定义 strict trace，不能与主 DSH + TorchPort 的诊断质量直接排名。

## 8. 改进建议

### 8.1 把迁移流程固定为五个阶段

1. **结构检查：** 模块图、state-dict key、alias、shape、dtype、参数数量。
2. **最小 smoke：** 只证明 RN，不在此阶段声称数值成功。
3. **阶段对齐：** 自动保存首个不一致层及其输入/输出，不允许先改下游算子。
4. **公共 evaluator：** 使用统一 fixture、容差和重复运行；记录 RN/NA/max/mean/rounds。
5. **可选 strict：** 只有研究问题需要时才做；与公共 Rounds 分开记录。

### 8.2 RAFT 必须保存的最小诊断点

- `fnet.feature.0` 和最终 feature；
- raw correlation volume；
- context encoder 输出；
- 第 1、6、12 次迭代的 hidden state、delta flow、mask；
- `flow_low`、`flow_up`。

每次只报告首个超阈值节点、max/mean/mismatch count 和输入是否已经一致。这样能避免产生大量难以解释的图和指标。

### 8.3 强制环境隔离

- Codex、pure DSH、DSH + TorchPort 使用不同 workspace、conda env 和 Jittor cache。
- TorchPort target subprocess 不得复用主实验的 `jittor_core` cache。
- 记录 Python、Jittor、CUDA、cuDNN、GPU、关键环境变量和源码哈希。
- evaluator 至少重复两次；若输出不同，先判定 reproducibility failure，再处理 NA。

### 8.4 分开统计修复类型

每轮增加 `repair_class`：

- `semantic`：模型、数据、权重、训练协议；
- `numeric`：融合边界、归约树、内核选择；
- `environment`：编译器、缓存、驱动、SSH；
- `evaluator`：dtype、序列化、舍入和文件格式。

APT 的 ULP 修复、CLIP 的缓存污染和 RAFT 的 Welford 修复不应被压缩成同一种 “Round”。

### 8.5 设置 strict 的停止条件

若同时满足以下条件，应允许结论为“公共通过，strict 后端受阻”：

1. first divergence 已稳定定位；
2. 上游输入逐位一致；
3. 剩余差异来自库内闭源 kernel；
4. 无法从公开 API 选择参考使用的同一 engine；
5. 已记录尝试过的算法和误差下界。

这比继续堆积任务专用 CUDA 仿真更可复现，也更诚实。

## 9. 最终审查意见

- **CLIP：** 三种方法都证明了基础 PyTorch→Jittor 结构迁移能力；结果不能证明 TorchPort 提升了修复效率，因为所有方法都在 Round 0 通过。
- **APT：** 真正的迁移对象是完整训练轨迹，而非单次 forward。Codex 已完成并通过，但 DeepSeek 会话缺失，不能做方法排名。
- **RAFT：** 是当前最有区分度的实验。Codex 通过系统化五轮修复达到公共 NA；pure DSH 因定位不收敛和结果不稳定被停止；DSH + TorchPort 用一轮公共修复通过，并显著提高了 strict 诊断精度，但仍不能复刻 4 个 cuDNN 闭源内核；V4 Flash 以更复杂的 compensated arithmetic 在三轮后通过公共容差。

最重要的可复用结论是：

> PyTorch → Jittor 迁移不能只验证算子公式和最终输出。对训练任务要验证数据与状态轨迹；对数值敏感的循环模型要验证 eager/fusion 边界、归约树和后端 kernel。只有把结构一致、可运行、公共数值一致和严格中间一致分开，才能准确解释代理究竟解决了什么。
