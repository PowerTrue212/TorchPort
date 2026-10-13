# held-out v1 阶段报告（2026-10-10）

## 1. 结论

1. 四个 COD 仓库（CamoFormer、BiRefNet-COD、PlantCamo、VSCode-v2-T）都由 DeepSeek V4.1 flash 在 DSH
   中完成了原生 Jittor 迁移。四个实现在全测试集上的任务指标与 PyTorch 的差都在 1e-7 量级。
2. 按逐层验收协议 v3：2 个通过，1 个允许提交，1 个失败。失败的是 PlantCamo，有 17 处 LayerNorm eps
   写反。这个错误让任务指标只变了 2e-7，只有逐层比较能发现。
3. 这四次运行的反馈条件、协议版本、访问控制各不相同，只能作为四个独立案例。**TorchPort 反馈对 Agent
   迁移能力的影响，目前还没有可用的对照数据。**
4. 这一阶段的主要产出是验收协议本身：从固定容差（v1）到 v2，再到现在冻结的 v3，并修掉了采集、
   比较、定位中的一系列缺陷（见第 5 节）。

## 2. 验收方法摘要

每一层比较相对误差 rel(T, R) = ‖T − R‖ / ‖R‖，R 是 PyTorch fp32（关闭 TF32）参考，T 是 Jittor 的一次运行。

- D：PyTorch 换几种不改变数学含义的运行方式（cudnn benchmark、fp64、CPU、channels_last、批量 2 且样本不同）后，
  这一层所有运行两两之间的最大 rel。
- N_env：同一 Jittor 环境中，用随机数据单独重复运行常见算子得到的运行间最大 rel，不读候选代码。
  3080 Ti 上矩阵乘、LayerNorm、softmax 为 0，卷积约 1.8e-7，全局池化与全局求和约 2e-6。
- 比值 = max_i rel(T_i, R) / max(D, N_env)，候选 3 次运行每次都参与。全部层最大比值 ≤ 2 通过，
  2–4 允许提交（以任务指标为准），> 4 失败。
- 输出张量另做逐元素比较；模块构造参数（eps、padding、align_corners 等）逐项比对，只作诊断。

判定线是看过 7 个已知对错的候选后定的：正确实现最大比值 ≤ 2.99，错误实现 ≥ 11。后续仓库只能检验，
不能再用来调整。

## 3. 四个仓库

### 3.1 运行概况

| | CamoFormer | BiRefNet-COD | PlantCamo | VSCode-v2-T |
| --- | --- | --- | --- | --- |
| 运行 | 10-05 试点 | none-r0 | full-r0 | full-r0 |
| 反馈条件 | 工具未接远端 GPU | diagnostic（不给位置） | localized | localized |
| 运行时协议 | v1 固定容差 | v2 | v2 | v3 |
| 墙钟 | 约 63 分钟 | 2h44m | 2h40m | 67 分钟 |
| 模型步数 | 219 | 333 | 234 | 390 |
| token | 6060 万 | 9150 万 | 5060 万 | 7950 万 |
| 费用 | 2.58 元 | 6.15 元 | 3.87 元 | 2.66 元 |
| TorchPort 调用 | inspect 1 / run 1 / verify 1 | inspect 1 / run 1 / verify 5 | inspect 1 / run 2 / evaluate 1 / verify 5 / trace 1 / localize 1 | inspect 1 / run 3 / evaluate 2 / verify 3 |
| 单次 verify 耗时 | — | 15–25 分钟（提速前） | 1–2 分钟 | 约 1 分钟 |

合计约 2.8 亿 token、15.3 元，99% 以上是缓存读取。

### 3.2 迁移结果

| | CamoFormer | BiRefNet-COD | PlantCamo | VSCode-v2-T |
| --- | --- | --- | --- | --- |
| 参数对齐 | 1100/1100 | 711/711 | 1278/1278 | 783/783 |
| 运行时验收 | fail（TF32 误判） | fail，1.021 | fail，20.3 | pass，1.074 |
| v3 复验 | pass，0.65 | 允许提交，2.99 | fail，11.1 | pass，1.074 |
| 实际状态 | 正确 | 正确 | 17 处 eps 写反；修正后 pass 1.97 | 正确 |
| E2 图片数 | 6473（4 个集） | 326（CHAMELEON、CAMO） | 250 | 6397（3 个集） |
| E2 指标最大差 | ~1e-7 | 8.5e-8 | 2.0e-7（修正后 5.6e-8） | 3e-7 |
| uint8 逐像素最大差 | 1 | 1 | 1 | 1 |
| 128 阈值翻转比例 | < 1e-6 | ≤ 5.3e-8 | 1.5e-7 | ≤ 6.0e-8 |
| Jittor 相对 PyTorch 推理耗时 | 约 4.2 倍（V100） | 约 10 倍 | 约 1.3 倍 | 约 10 倍 |

证据：`results/<repo>-<run>/`，校准报告在 `results/v3-calibration/`。CamoFormer 证据在
`migrations/CamoFormer/outputs/`。

### 3.3 各次运行的偏差

| 运行 | 偏差 |
| --- | --- |
| CamoFormer | TorchPort 工具未接远端 GPU；运行时协议为固定容差 |
| BiRefNet | 续用旧会话，cwd 为仓库根目录，AGENTS.md 被注入；Agent 读了 TorchPort 源码；E2 只跑 326 张 |
| PlantCamo | Agent 在远端读取了 verify 报告 JSON（localized 条件下内容本可见，无额外信息）；trace 因远端磁盘写满失败 |
| VSCode-v2 | 无越界访问；agent_ssh 误拦 2 次（已修）；比较覆盖率 51%，550/561 层上限由 N_env 决定 |

## 4. 观察

1. **任务指标不足以判断实现是否正确。** PlantCamo 的 eps 错误在 250 张图上让 S-measure 只变了 2e-7，
   有差异的像素从 7.5e-6 升到 3.9e-5。逐层比较给出比值 11。
2. **Agent 会把实现错误归因给框架。** PlantCamo 的 Agent 收到了首次分歧 `backbone.norm1` 的定位，
   仍结论为"Jittor 数值上无法对齐"。模块构造参数比对上线后，这类错误能直接报出（25 处差异，其中被调用的 17 处正是 eps）。
3. **Agent 总结的框架问题大多不成立。** VSCode-v2 报了 5 条，复现后 1 条成立（`avg_pool2d` 窗口不能等于输入），
   2 条不成立，2 条 PyTorch 行为相同。BiRefNet 报的 5 条未复现。
4. **Agent 主要依靠自建对照。** 三次 held-out 运行中，Agent 都在远端搭了逐模块的 PyTorch 对照；
   VSCode-v2 第一次调用 verify 已在开跑 57 分钟后。这意味着 diagnostic 组也能自己补回部分定位能力。
5. **Jittor 运行间不确定的来源。** conv_opt=1 的卷积与全局空间归约用原子加法累加。全局池化运行间差约 2e-6，
   对 fp64 的误差是 PyTorch 的 10 倍；拆成两步 mean 或改用矩阵乘后逐位确定（P3 已记录）。
   conv_opt=0 在本机不可用是 cuDNN 8/9 混装所致，补 LD_LIBRARY_PATH 后单个卷积与 PyTorch 逐位相同，
   但整网误差反而增大（patch_embed3 约 5.9e-5），原因未查明。

## 5. TorchPort 本阶段改动

| 改动 | 效果 |
| --- | --- |
| 比较不再调用多线程 BLAS 求范数 | BiRefNet 单次比较 1601 s → 11.7 s，685 个张量判定不变 |
| hook 记录强制复制 | 修掉 CPU 变体被原地运算改写的问题（35–46 个张量的 D 被撑到 0.5–1.8） |
| 采集版本进入 oracle 身份 | 采集代码改动后旧 oracle 自动重建 |
| 协议 v3 | 每次运行都参与判定、候选波动不放宽上限、三档结论、输出逐元素比较、覆盖率报告 |
| N_env 自动标定 | 与候选无关，按环境缓存，形状取 source 最大全局归约 |
| 模块构造参数比对 | 直接报出 eps / align_corners 等不一致 |
| 定位（trace） | 只记录首次分歧附近的模块，产物 837M/403M → 172M/59M；内置层进入单层比较；单层比较按算子取 N_env；磁盘不足时拒绝启动 |
| 访问控制 | 工作区放在仓库外；agent_ssh 拦截访问证据目录的命令并记日志；输入与权重预置在 Agent 远端目录 |
| P3 条目 | 5 条（observed）：PVT eps 默认值、全局归约不确定、UpsamplingBilinear2d 默认 align_corners、avg_pool2d 窗口限制、Jittor 归约不确定 |

测试 160 个通过。

## 6. 已知局限

1. N_env 取全局最大值。D 很小的模型（如 VSCode-v2）上，判定实际是"误差 < 约 2 倍 N_env"，
   eps 写反这类温和错误只会落到"允许提交"。
2. N_env 在单个算子上测得，误差经下游层放大时覆盖不到（BiRefNet 全局池化后接 BN，比值接近判定线）。
3. 比值 4 到 60 之间没有已知错误的候选，这个区间内判定是否可靠未经检验。
4. E1 只用一张输入图；参与比较的层占 51%–94%；大张量压缩后估计误差。
5. 每个条件每个仓库只有 1 次运行。

## 7. 建议补跑

按优先级：

| # | 内容 | 目的 | 估计成本 |
| --- | --- | --- | --- |
| 1 | 温和错误对照：在 VSCode-v2 和 CamoFormer 正确实现上各注入 3–4 个小错误（单处 eps、单处 align_corners、插值模式、BN eps），只跑 verify | 检验比值 4–60 区间，尤其 N_env 主导时 v3 能否拦住 | 远端约 1 小时，无 token 费用 |
| 2 | 多输入与重复：BiRefNet、VSCode-v2 正确实现各换 2 张输入、BiRefNet 重复 verify 3 次 | 判定是否稳定，BiRefNet 判 fail 的概率 | 远端约 1–1.5 小时 |
| 3 | 反馈对照：PlantCamo 与 VSCode-v2，none / full 各 3 次，v3 协议、新会话、相同预算 | 回答 TorchPort 反馈有无作用 | 12 次运行，墙钟约 15–30 小时，约 40–60 元 |
| 4 | BiRefNet E2 补 NC4K、COD10K | 补齐任务指标 | 远端约 17 小时，可后台 |
| 5 | BiRefNet Agent 声称的 5 条 Jittor 问题复现 | P3 | 约 30 分钟 |

第 1、2 项应在第 3 项之前：如果温和错误被放过，或正确实现频繁判 fail，对照实验的停止规则就不可靠。
第 3 项开跑前需要定 none 组的反馈内容（目前为 diagnostic：通过与否、阶段、聚合数值，不给位置）。
