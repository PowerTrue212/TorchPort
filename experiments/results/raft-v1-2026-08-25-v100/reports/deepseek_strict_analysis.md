# RAFT strict hardening：DSH+TorchPort 未通过原因与 Codex 对照分析

## 1. 结论摘要

DSH+TorchPort 没有通过 `operator-strict`，直接原因不是 RAFT 结构迁移错误，也不是最终 optical-flow 输出失效，而是：**源端与目标端实际使用了不同 CUDA/cuDNN 软件栈和不同卷积执行计划；strict 又要求 30 个中间张量逐层满足 `rtol=1e-5, atol=1e-6`。**

本轮实际环境是：

| 项目 | PyTorch reference | Jittor target |
|---|---|---|
| Framework | PyTorch 2.6.0+cu124 | Jittor 1.3.11.0 |
| CUDA toolchain | 12.4 | 11.3 |
| cuDNN runtime | 9.1.0 (`90100`) | 8.2.0 (`8200`) |
| GPU | V100-SXM2-16GB | 同一张 V100-SXM2-16GB |
| 卷积接口 | cuDNN frontend/backend execution plan | legacy v7 algorithm API |

因此，当前 strict 实际测试的不只是“PyTorch 代码是否正确迁移到 Jittor”，还测试了“cuDNN 9.1 frontend engine 是否能被 cuDNN 8.2 legacy 路径逐层数值复刻”。这一要求明显强于主实验的最终输出一致性目标。

最重要的对照结论是：**Codex 也没有解决 strict。**在同一份 TorchPort source trace 上直接测量最终 Codex 实现：

| 实现 | `feature.0` max abs | mean abs | strict 失配元素 | 失配比例 |
|---|---:|---:|---:|---:|
| Codex | 6.0201e-6 | 7.4012e-7 | 10,048 | 1.2777% |
| DSH+TorchPort hardening | 3.5763e-6 | 4.1385e-7 | 797 | 0.1013% |

DSH+TorchPort 在首处分叉上比 Codex 更接近 reference；Codex 的成功是“最终输出进入公共 evaluator 容差”，不是“中间算子逐层一致”。

## 2. strict 失败到底发生在哪里

`compare_strict.json` 的首个失败是：

```text
tensor: feature.0
position: 2
shape: [1, 256, 48, 64]
rtol: 1e-5
atol: 1e-6
max_absolute_error: 3.5762786865234375e-6
mean_absolute_error: 4.1384802125972914e-7
mismatched_elements: 797 / 786432
mismatch_ratio: 0.0010134379069010417
cosine_similarity: 0.9999999999998024
```

这不是大范围错误。797 个失败元素中，reference 绝对值的中位数只有约 `0.0155`，90% 小于约 `0.0479`。这些位置的 allclose 允许误差大约只有 `1.0e-6` 到 `1.5e-6`；失败元素误差中位数约 `1.36e-6`。也就是说，strict 主要被**接近零的小幅值特征**卡住，而不是出现了方向、尺度或结构性错误。

“28/30 张量失败”也不能解释成 28 个独立根因：输入两张量通过，`feature.0/1` 首先失败；之后 RAFT 的相关性查询和 12 次递归更新会传播并累积这个微小偏差。最终 `output` 有 113,108 个 strict 失配元素，主要是首处分叉和后续卷积差异经过递归放大的结果。

## 3. 为什么相同 FP32 卷积仍会不同

### 3.1 FP32 运算顺序决定最后几位

浮点加法和乘法不满足结合律，卷积采用不同 tile、通道分块、归约树、FMA 收缩或 Winograd 变换时，即使数学公式相同，最后若干 bit 也可能不同。PyTorch 官方明确说明，数学等价的浮点计算不保证 bitwise identical，跨 release 和平台也不保证逐位一致。[PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)

### 3.2 “确定性”不等于“跨实现相同”

确定性只说明同一个实现、同一执行计划重复运行得到相同结果。它不保证两个不同框架、不同 cuDNN 版本或不同 engine 的输出相同。PyTorch 的复现文档也区分了关闭 benchmark、选择确定性算法与跨环境复现，并明确指出跨 release/platform 不保证完全复现。[PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)

NVIDIA 对 cuDNN 的保证同样限定为“给定版本、相同 GPU 架构下的大部分 routine 在重复运行中一致”，没有承诺 cuDNN 8.2 和 9.1、或不同 execution plan 之间逐位一致。[cuDNN Reproducibility](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-870/developer-guide/index.html#reproducibility-determinism)

### 3.3 两端没有使用同一卷积选择机制

PyTorch 2.6 的 `Conv_v8.cpp` 构造 cuDNN operation graph，从 heuristics 获取 engine config，再构造和缓存 execution plan；engine config 还会根据 determinism、down-conversion 和 TF32 numerical notes 过滤。[PyTorch 2.6 Conv_v8.cpp](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cudnn/Conv_v8.cpp)

当前 Jittor 1.3.11 的安装源码则调用：

```text
cudnnFindConvolutionForwardAlgorithmEx(...)
或 cudnnGetConvolutionForwardAlgorithm_v7(...)
然后 cudnnConvolutionForward(..., algo, ...)
```

Codex 设置 `jt.cudnn.set_algorithm_cache_size(0)` 后，Jittor 会跳过 timing-based `Find`，改用 legacy v7 heuristic；这能让 Jittor 自身稳定，却不会自动选中 PyTorch cuDNN 9.1 frontend 的同一 engine 和 knobs。

NVIDIA 文档说明，一个 operation graph 可以有多个 engine，每个 engine 还有可调 knobs；heuristic 只负责排序候选。不同 engine 可能具有不同 numerical notes 和归约实现。[cuDNN Frontend Graph API](https://docs.nvidia.com/deeplearning/cudnn/v1.12.0/developer/graph-api.html)

## 4. Codex 做法与 DSH+TorchPort 做法的真实差异

### Codex 的优化目标

Codex 的路径是以公共 evaluator 为中心：

1. 关闭 Jittor timing-dependent cuDNN Find，保证重复运行稳定；
2. 用固定 CUDA reduction 实现 InstanceNorm；
3. 显式展开 9 路 softmax 与 convex upsample sum；
4. 在 grid 坐标变换处插入 `stop_fuse()`，保存 PyTorch eager 舍入边界；
5. 调整 eval BatchNorm 的减、除、乘、加顺序。

这条路线最终实现：公共 evaluator `RN=true, NA=true`，两次结果逐位一致。但其 InstanceNorm reduction、输入归一化和部分卷积路径并未追求完整复刻 PyTorch 内核，因此不应推断为 strict 成功。

### DSH+TorchPort 的优化目标

DSH+TorchPort 在 hardening 阶段继续追逐算子级轨迹：

1. 用 PTX `div.rn.f32` 修正除法语义；
2. 复刻 PyTorch Welford reduction 和 `rsqrtf`；
3. 在 PyTorch eager kernel 边界插入 `stop_fuse()`；
4. 复刻 warp softmax 归约树；
5. 将 GRU 非对称卷积分解为 5 次移位 1×1 卷积。

这些工作使 `feature.0` strict 失配从 5,751 降到 797，相比 Codex 的 10,048 也更好。TorchPort 在这里确实提供了额外价值：它把误差定位到首个中间阶段，而不是只依赖最终 flow。

### 为什么 Codex 最终 flow 反而略好

在 strict 输出容差下：

| 实现 | output max abs | mean abs | strict 失配元素 | 比例 |
|---|---:|---:|---:|---:|
| Codex | 1.2374e-4 | 6.7252e-6 | 111,868 | 28.4495% |
| DSH+TorchPort | 1.4865e-4 | 6.8013e-6 | 113,107 | 28.7646% |

这不与 `feature.0` 结果矛盾。RAFT 是递归系统，后续算子的误差可能累积，也可能部分抵消。Codex 在首个 feature 上较差，但其后续整体误差轨迹恰好让最终 flow 略接近；DSH 在首个 feature 上明显更好，但后续自定义 GRU/卷积路径贡献了新的微小差异。最终输出误差不能反推出每一层都更准确。

## 5. 对“cuDNN 闭源，因此不可修复”的评估

### 已被证据支持的部分

- 通过普通 Jittor cuDNN 8.2 legacy convolution 无法复刻 PyTorch cuDNN 9.1 frontend 的全部目标 shape；
- 手写顺序 FMA、不同累加顺序和已有 Winograd 尝试未达到 strict；
- 剩余误差符合不同 cuDNN engine/tile/reduction 路径导致的末位差异特征；
- 继续猜测闭源 kernel 的内部归约树，收益很低。

### 表述过强的部分

“绝对不可修复”尚未被证明。cuDNN backend API允许指定 `engine_id + knobs` 创建自定义 execution plan，也支持查询和序列化 execution plan。[NVIDIA Custom Execution Plan](https://docs.nvidia.com/deeplearning/cudnn/latest/utilities/custom-execution-plan.html) [cuDNN Backend Execution Plan](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html)

本次对 PyTorch reference 开启 cuDNN frontend logging 后，已经获得关键 shape 的实际计划：

| 目标卷积 | PyTorch cuDNN 9.1 execution plan |
|---|---|
| `layer3[0].conv1`, 96→128, 3×3, s2 | `ConvFwd_eng31_k2=2_k4=2_k5=3_k6=2_k7=1` |
| `encoder.convc1`, 324→256, 1×1 | `ConvFwd_eng58_k25=2` |
| GRU 384→128, 1×5 | `ConvFwd_eng31_k2=2_k4=2_k5=3_k6=2_k7=1` |
| GRU 384→128, 5×1 | `ConvFwd_eng4` |

这说明仍存在一条尚未充分验证的路线：**在 Jittor 自定义 op 中直接链接与 reference 相同的 cuDNN 9.1，使用 backend/frontend API按上述 engine 和 knobs 构造固定 execution plan。**最终 forward 仍是 native Jittor 调度，不需要调用 PyTorch，也不需要读取 reference 输出。

因此，准确结论应写为：

> 在当前 Jittor 1.3.11 + CUDA 11.3 + cuDNN 8.2 legacy 路径，以及已经实施的手写 kernel 方案内，strict 无法继续收敛；要进一步验证，需要让目标端使用与 PyTorch reference 相同的 cuDNN 9.1 frontend execution plan。当前结果不能证明数学或工程上的绝对不可实现。

## 6. 实验设计层面的影响

主实验冻结了模型、checkpoint、输入、GPU、FP32 和最终 evaluator，但没有冻结两端相同的 CUDA/cuDNN runtime。对最终 `flow_up` 的 `1e-4/1e-4` 比较，这一设计合理；对于中间算子的 `1e-5/1e-6` strict 比较，runtime 版本已经成为一个额外变量。

因此建议把结论分层：

1. **主实验结论有效**：Codex 和 DSH+TorchPort 都保持了最终 RAFT 推理结果；DSH+TorchPort 的公共修复轮数为 1，Codex 为 5。
2. **strict 结论有效但含义不同**：两种实现都未达到跨框架、跨 cuDNN 版本的逐阶段高精度一致；DSH+TorchPort 在首处分叉上显著优于 Codex。
3. **不能把 strict=false 写成迁移失败**：它与公共 evaluator 的任务级成功定义不同。
4. **不能把 28/30 写成 28 个迁移错误**：多数是首处分叉经 RAFT 递归传播后的结果。

## 7. 建议的下一步验证

### A. 最有价值：cuDNN 9.1 plan replay

只针对四种剩余 shape 编写 Jittor custom op：

1. 链接 PyTorch wheel 中同版本的 cuDNN 9.1 shared libraries；
2. 用 cuDNN backend/frontend 构建相同 operation graph；
3. 固定上述 engine id、knobs 和 workspace；
4. 先逐卷积与 source 输出比较，再接回完整 RAFT；
5. 连续运行两次 strict、fp32 和公共 evaluator。

若逐卷积仍不一致，再检查 tensor alignment、stride、workspace、math type、numerical notes 和 cuDNN frontend tag，而不是继续猜测手写累加顺序。

### B. 增加细粒度 trace

当前正式 strict trace 只包含输入、`feature.0/1`、每轮 flow 和输出；“剩余四个卷积”的结论主要保存在 agent 的 strict round 文字记录中。下一轮应正式保存：

```text
layer3.0.conv1 pre/post
layer3.0.norm1 post
fnet.conv2 post
motion_encoder.convc1 post
gru.convz1 post
gru.convz2 post
```

这样 certificate 可以直接证明每个剩余卷积，而不是依赖 agent 自述。

### C. 环境匹配对照组

另开 `RAFT-Strict-Stack`，让 PyTorch reference 与 Jittor target 使用同一 CUDA/cuDNN runtime，再运行 operator-strict。这个实验可以区分：

- 框架算子语义差异；
- cuDNN 版本/engine 差异。

不要修改当前第一轮 reference，否则会破坏与 Codex、pure DSH 的公平比较。

## 8. 最终判断

DSH+TorchPort 未通过 strict 的主要原因是**跨 cuDNN 版本、跨 API、跨 execution plan 的卷积累加顺序差异，被极严格的中间张量容差和 RAFT 递归结构放大**。DeepSeek 已经修复了可由公开语义复刻的大部分差异；其首个 feature 严格误差明显优于 Codex。

Codex 提供的启示是：先消除算法选择波动、控制 fusion 边界、复现关键 normalization/coordinate 语义，就足以通过任务级 evaluator。它没有证明 strict 可通过，也不能作为“DeepSeek 能力不足”的证据。

当前应如实保留：

```text
Codex: public pass, strict 未验证为通过；实测 feature.0 strict 不通过
DSH+TorchPort: public pass, fp32 certificate pass, operator-strict false
```

如果仍要追求 strict，下一步不应继续手写猜测闭源卷积，而应执行同版本 cuDNN 9.1 frontend plan replay。

## 9. 本地证据索引

- `results/dsh_torchport/traces/compare_strict.json`
- `results/dsh_torchport/traces/compare_fp32.json`
- `results/dsh_torchport/traces/strict_rounds.jsonl`
- `results/dsh_torchport/traces/reference_cudnn_frontend.log`
- `workspaces/codex/repo/submission.py`
- `workspaces/dsh_torchport/repo/submission.py`

