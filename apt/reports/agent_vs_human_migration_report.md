# APT 从 PyTorch 迁移到 Jittor：Codex、DSH 与人工迁移过程对比报告

**报告日期：** 2026-09-02  
**研究对象：** APT（Additive Prompt Tuning）持续学习项目的 PyTorch → Jittor 迁移  
**结论状态：** 暂定的两方比较；APT 的 DSH 会话与实验产物尚未取得，因此三方结论不完整

## 1. 摘要

本次审计能够完整还原 Codex 的 APT 迁移会话、修改策略、修复轮次和冻结评测结果，也能够从 `D:\大学\新芽` 中的源代码、人工迁移代码、报告和日志重建人工迁移过程。但在本机 DSH 会话索引和所有可读取的 DSH 压缩会话中，未发现 APT 迁移会话；V100 当前离线，无法进一步核验仅存在于远端的 `workspaces/dsh` 或 `results/dsh`。因此，本文不会用 CLIP 或 RAFT 的 DSH 记录替代 APT 证据。

目前可以得到的主要判断如下。

1. **Codex 迁移更接近“冻结协议下的任务专用重实现”。** 它新增一个约 31.6 KB 的自包含 `jittor_apt.py` 和一个薄包装 `submission.py`，不大面积修改原仓库；在同一 V100、固定 checkpoint、固定任务顺序和统一 evaluator 下，完成 10 个任务、每任务 30 epoch 的从头训练，最终 `RN=true`、平均准确率 `89.1600037%`、遗忘率 `3.9666655%`，与 PyTorch 参考准确率仅差 `7.63×10⁻⁶` 个百分点。
2. **人工迁移更接近“按原工程结构逐模块改写”。** 它在 13 个共同文件中约增加 461 行、删除 500 行，并新增 `models/jimm.py`，修改分散在数据、ViT、模型封装、优化器、训练器和启动脚本中。这种方式更便于人理解框架 API 的对应关系，也更接近可维护的工程迁移，但修改面更广，调试和复现实验的状态更多。
3. **两者当前成绩不能直接排名。** 人工结果来自 Jittor 1.3.10、CUDA 11.8、`sm_89` GPU，使用 seeds 1/2/3 和 `ColorJitter`；Codex 来自 Jittor 1.3.11、V100 `sm_70`、冻结的单一参考协议，并复现了 ImageNet AutoAugment。硬件、随机种子、增强、代码版本和 evaluator 均不完全相同。
4. **Codex 的优势主要在协议控制和审计性，人工迁移的优势主要在结构透明和迁移知识表达。** Codex 有会话、轮次、固定哈希、外部 evaluator、断点和预测级产物；人工报告解释了每类 API 为什么要改，并给出了三随机种子的趋势，但源码—日志—报告之间没有不可变 commit/hash 绑定。
5. **APT 的 DSH 基线目前是缺失数据，不是失败结果。** 在补齐其会话、最终仓库和 evaluator 输出前，不能回答“Codex 是否优于 DSH”。

## 2. 证据范围与方法

### 2.1 Codex 证据

- 本地 Codex 原始会话：  
  `C:\Users\31561\.codex\sessions\2026\08\23\rollout-2026-08-23T23-42-12-01a02f49-5f69-7ce0-ad38-e80de3f70126.jsonl`
- 会话标题：`apt 迁移到 jittor`
- 已在此前在线核验的 V100 产物：`workspaces/codex/repo`、`results/codex`、统一 evaluator 输出和 NPZ 结果。
- 本地冻结实验协议：`D:\科研\TorchPort\apt\experiment.json`、脚本、模板与测试。

会话是约 14.7 MB 的逐事件 JSONL，能够定位用户指令、工具调用、代码补丁、评测启动、失败原因和第 1 轮修复。

### 2.2 DSH 证据

本机 DSH 会话索引位于：

`C:\Users\31561\.dsh\storages\session_projcache.json`

压缩会话位于：

`C:\Users\31561\.dsh\sessions\--D-~79D1~7814-TorchPort--\*.jsonl.zstd`

已检查的 5 个会话分别属于 CLIP、CLIP+TorchPort、RAFT、RAFT+TorchPort 和 DeepSeek V4 RAFT。对 APT 项目名、实验标识和 `/root/data-fs/APT` 等关键词的全文检索，只找到连接文档或其他实验偶然提到该路径，没有找到 APT 迁移指令、代码补丁、修复轮次或结果。故本报告把 DSH 标记为 **未观测（missing evidence）**，而不是“未通过”。

### 2.3 人工迁移证据

- PyTorch 源码：`D:\大学\新芽\apt`
- 人工 Jittor 代码：`D:\大学\新芽\APT-Jittor`
- 迁移报告：`任务三-代码迁移报告.docx`
- 迁移与验证报告：`任务四-基于 Jittor 框架的 APT 持续学习模型迁移与验证.docx/.pdf`
- 新版日志：根目录 `seed1.log`、`seed2.log`、`seed3.log`
- 旧版日志：`APT-Jittor\seed1.log`、`seed2.log`、`seed3.log`
- 汇总结果：`comparison_results\alignment_report.md`、`alignment_summary.json`

人工过程没有会话级记录，因此只能根据报告陈述、源码差异、日志内容和文件时间重建；不能把文件时间跨度解释成人工工时。

## 3. Codex 会话与修改内容

### 3.1 会话时间线

| 阶段 | 行为与结果 |
|---|---|
| 初次准备 | ZIP 校验通过，但 bootstrap 生成源码树哈希为 `6235c950…`，`experiment.json` 声明为 `8ebe18b8…`。Codex 遵守冻结协议，在创建工作区之前退出，修复轮次保持 `0/8`，没有绕过校验。 |
| 用户修正协议后重启 | 2026-08-24 00:27（北京时间）开始继续。约 11 分钟后已形成主要 Jittor 实现。 |
| 轮次 0，实施与预检查 | 完成模块实现、静态检查、最小烟测、训练参数与增强行为校正，然后启动统一 evaluator。 |
| 轮次 0，完整训练 | 10 个任务 × 30 epoch，耗时 `13,673.215 s`，约 3 小时 47 分 53 秒。模型和预测有效，但 evaluator 因最终矩阵均值的 float32 舍入差 `1.129×10⁻⁵`，刚好超过 `1×10⁻⁵` 容差而拒绝。 |
| 轮次 1 | 仅修正结果汇总时的浮点表示，利用 task-10 checkpoint 恢复并重评，耗时约 `48.58 s`；最终通过。已用修复轮次 `1/8`。 |

从重启到最终通过的会话墙钟时间约 9 小时 5 分钟，但其中包含环境探测、编译、烟测、监控和等待；它不等于纯编码时间，也不能直接与人工劳动小时比较。

### 3.2 Codex 的代码策略

Codex 没有把原仓库的每个模块逐一改写，而是在原项目中增加：

- `jittor_apt.py`：约 31,555 字节的自包含实现；
- `submission.py`：约 751 字节的统一入口包装。

主要修改/实现内容包括：

1. **数据路径和预处理**
   - 使用 pickle、NumPy 和 Pillow 直接加载 CIFAR-100；
   - 在 PIL 层复现 torchvision 的 ImageNet AutoAugment policy；
   - 显式校准随机数抽取顺序，避免同一种子下增强策略顺序漂移。
2. **模型与权重**
   - 在 Jittor 中实现 ViT-B/16：输入 224、隐藏维度 768、12 层、12 个注意力头、196 个 patch；
   - 从冻结实验提供的 NPZ checkpoint 加载预训练权重；
   - 实现 APT prompt token、全局 prompt 的 EMA 合并和余弦分类器。
3. **训练协议**
   - 只优化 prompt 和分类器，预检查中移除了 `clf_norm` 的可训练参数；
   - 使用 Jittor Adam 与 cosine schedule，并修正 scheduler 的 epoch 偏移；
   - 执行 10 个增量任务、每任务 30 epoch；每个任务后计算准确率矩阵。
4. **运行与恢复**
   - 记录 `progress.npz`，支持按任务恢复；
   - 最终保存 `accuracy_matrix.npy` 与 `predictions.npz`；
   - 最终提交代码不导入 PyTorch。
5. **兼容性修复**
   - PIL `Solarize` magnitude 类型适配；
   - 调整增强随机数调用顺序；
   - 修正 scheduler 步数；
   - 用精确的 float64 任务准确率计算与最近 float32 ULP 选择，处理 evaluator 的边界舍入。

### 3.3 Codex 最终结果与可信度

| 指标 | Codex Jittor | PyTorch 参考 | 解释 |
|---|---:|---:|---|
| RN | true | — | 完整 forward、训练和结果保存可运行 |
| 平均准确率 | 89.1600036621% | 89.16% | 差 `7.6293945×10⁻⁶` 个百分点 |
| 遗忘率 | 3.9666655064% | 对应冻结参考 | 通过统一 evaluator |
| 修复轮数 | 1 | — | 第 1 轮只处理结果汇总浮点边界 |
| 最终准确率 | 89.16% | 89.16% | 完全相同的汇总值 |

额外的产物级核验证据：

- targets 和 task IDs 与 PyTorch 完全一致；
- 10,000 个样本上的预测一致率为 95.4%；
- logits 并不相同，最大绝对误差约 2.9091；
- 准确率矩阵最大差约 2.70 个百分点；
- `results/codex` 与其 `run` 子目录中的最终产物哈希一致；
- `progress.npz` 包含已完成任务、矩阵、prompt、EMA prompt、分类器权重等实际训练状态。

这些证据说明 Codex 不是复制 `logits_ref` 或准确率参考文件来通过评测，而是训练了独立实现。不过，轮次 1 对最终矩阵元素做了一个 ULP 级表示调整，暴露出 evaluator 使用 float32 均值和严格边界容差的弱点。其数值仍在逐任务容差内，且没有改变权重和预测，但后续实验应修复 evaluator 或明确冻结这一行为。

## 4. DSH 会话审计结果

### 4.1 当前能确认的事实

- 本地 DSH 会话索引中没有标题或元数据指向 APT；
- 对 5 个完整压缩会话做关键词检索，没有找到 APT 的迁移任务、补丁、训练日志或 evaluator 结果；
- 找到的 `/root/data-fs/APT` 只是连接文档/路径的偶然引用，相关会话实际执行的是 RAFT；
- V100 当前关闭，无法检查远端是否另存有未同步会话或 `results/dsh`。

### 4.2 研究解释

本报告不能把“没有本地记录”解释为 DSH 技术失败。更准确的编码是：

| 字段 | 当前值 |
|---|---|
| DSH APT session observed | false |
| DSH APT repository observed | false |
| DSH APT evaluator result observed | false |
| Failure reason | missing evidence / remote unavailable |
| 可用于方法比较 | 否 |

这一区分很重要：失败率研究中，“代理运行失败”“实验未运行”“记录遗失”“当前无法访问”必须是四个不同状态。

## 5. 人工迁移过程与修改内容

### 5.1 总体策略

人工迁移采用“保持原工程分层、逐模块替换框架 API”的策略。与 Codex 的单文件任务实现不同，它保留了 `dataloaders`、`learners`、`models`、`utils`、`trainer.py` 和 `run.py` 的边界，让每个 PyTorch 概念在 Jittor 中有清晰对应。

在 34 个源侧相关文件与 18 个目标侧相关文件的过滤比较中：

- 13 个同名文件发生变化；
- 约增加 461 行、删除 500 行；
- 新增 `models/jimm.py`；
- 17 个源侧文件未出现在 Jittor 目录，主要是其他数据集配置/脚本、划分 YAML、`models/moco.py` 和依赖文件。

这说明人工版本不是完整保留原仓库全部能力，而是聚焦 CIFAR-100 的 APT 主路径。

进一步的逐 hunk 分类审计表明：排除 9 个纯注释/格式 hunk 后，56 个功能性 hunk 共包含 1,089 行增删，其中直接 API 映射仅占 160 行（14.7%）；其余 85.3% 属于依赖/能力重实现、框架语义适配或工程适配。即使再排除 4 个只删除未支持数据集的范围收窄 hunk，并剔除空白、纯注释和 docstring，直接 API 映射仍只占 598 个代码改动行中的 106 行（17.7%）。完整口径与逐 hunk 明细见 `apt/reports/apt_human_diff_audit.md` 和 `apt_human_diff_hunks.csv`。

### 5.2 文件级修改范围

| 文件 | 近似增/删行 | 主要作用 |
|---|---:|---|
| `models/zoo.py` | +205 / -74 | Jittor 模型创建、权重导入、位置编码适配、余弦分类头、`clf_norm` |
| `dataloaders/dataloader.py` | +7 / -181 | 将数据流程收窄到 CIFAR-100/Jittor loader |
| `learners/default.py` | +70 / -57 | 训练、优化器和评估 API 迁移 |
| `models/vit.py` | +56 / -36 | `forward→execute`、注意力与 prompt 拼接、Jittor 算子 |
| `learners/prompt.py` | +43 / -24 | prompt 学习和 Jittor 优化流程 |
| `trainer.py` | +12 / -47 | Jittor 运行与设备逻辑 |
| `run.py` | +24 / -11 | 参数、随机种子和训练入口 |
| `dataloaders/utils.py` | +28 / -60 | Jittor transforms，增强策略变化 |
| `models/jimm.py` | 新增 | 替代 timm：PatchEmbed、DropPath、trunc_normal、输入卷积适配 |

### 5.3 人工报告记录的关键迁移问题

1. **基础 API 对应**
   - `torch`/`torch.nn` 替换为 `jittor`/`jittor.nn`；
   - `forward` 改为 `execute`；
   - PyTorch 的 `zero_grad → backward → step` 改为 `optimizer.step(loss)`。
2. **避免 Jittor 不兼容的原地写入**
   - Attention 中 prompt token 的切片原地赋值改为 `jt.concat`；
   - 屏蔽旧类别 logits 的原地写入改为 `jt.where`。
3. **预训练权重迁移**
   - 将 `.pth` 权重转换为 Jittor 可加载格式；
   - 自定义 loader，处理 key 映射、位置编码插值和分类头形状不匹配。
4. **timm 替换**
   - 新增 `jimm.py`，自行实现 ViT 所需的小型组件；
   - `trunc_normal` 中使用 NumPy/float64 中转实现 Jittor 缺少的 `erfinv` 路径。
5. **数值稳定性修复**
   - 在 cosine classifier 前增加 `LayerNorm(768)`；
   - 权重归一化时加入 `1e-8`，解决特征尺度异常导致的巨大 loss。
6. **数据增强变化**
   - 以 `ColorJitter` 代替 PyTorch 版本使用的 AutoAugment；
   - 人工报告也把它列为与 PyTorch 仍有准确率差距的潜在原因。

### 5.4 人工实验结果及版本问题

`comparison_results` 与根目录 2026-04-27/28 日志给出的较新结果为：

| 实现 | seed 1 | seed 2 | seed 3 | 平均 ± 标准差 |
|---|---:|---:|---:|---:|
| PyTorch | 89.1458 | 87.9271 | 89.0938 | 88.7222 ± 0.5627 |
| 人工 Jittor（新版日志） | 87.9167 | 86.9583 | 87.4583 | 87.4444 ± 0.3914 |
| 差值 | — | — | — | -1.2778 个百分点 |

但是目录内还存在另一组较旧日志：85.8854、83.6562、86.0521，均值约 85.1979%，与 PyTorch 相差约 -3.5243 个百分点；根 README 仍引用这组旧数字。较新报告引用 87.4444%，而当前若干源码文件的修改时间又晚于两批日志。由于没有 commit ID、源码树哈希或容器镜像把“某份代码”与“某次日志”固定绑定，当前只能确认这些结果曾被记录，不能证明当前目录代码原样重跑一定得到同一结果。

人工日志还显示：

- Jittor 1.3.10.0、CUDA 11.8、GPU 架构 `sm_89`，并非 Codex 使用的 V100 `sm_70`；
- 新日志的三个 seed 最终准确率完整，但较旧 seed 2/3 日志的 forgetting 汇总出现 `nan`；
- 报告中的速度比较缺少同机、同 batch、同实现版本的冻结证明，不宜用于 agent/人工效率排名。

## 6. Codex 与人工迁移的直接对比

| 维度 | Codex | 人工迁移 | 研究含义 |
|---|---|---|---|
| 迁移目标 | 严格完成冻结 APT/CIFAR-100 协议 | 保持原工程形态的 Jittor 移植 | 前者偏 benchmark，后者偏软件工程 |
| 修改形态 | 新增自包含实现 + 薄入口 | 13 个模块分散改写 + 新增兼容层 | 不能只按改动行数判断优劣 |
| 原源码侵入 | 低 | 高 | Codex 更容易隔离；人工更容易逐模块阅读 |
| 框架纯度 | 最终无 `torch` 导入 | 主流程迁为 Jittor | 两者都达到框架迁移目的 |
| 数据增强 | 主动复现 torchvision AutoAugment 与 RNG 顺序 | 改用 ColorJitter | 人工变化引入了协议差异 |
| 权重加载 | 直接适配冻结 NPZ | `.pth→.jt` + 自定义 loader/插值 | 两者输入工件不同 |
| 训练对象 | prompt + classifier；显式排除 `clf_norm` | prompt/classifier，且为稳定性引入可见的 `clf_norm` 设计 | 模型定义存在差异，需统一后再比较 |
| 调试方式 | 会话可追踪；训练前多次 parity 修正；外部 evaluator 驱动 1 轮修复 | 报告记录按错误现象逐项解决，但无逐操作会话 | Agent 的过程数据更适合自动统计 |
| 恢复能力 | task 级 `progress.npz` | 日志和脚本可重跑，未见等价冻结断点证据 | 长训练实验中 Agent 更抗中断 |
| 验证强度 | 同一 evaluator 检查矩阵、预测、targets、task IDs、禁用导入等 | 三 seed 指标和图表，缺少统一 evaluator | 二者验证目标不同 |
| 结果血缘 | 源码树、输入、产物、轮次均可追踪 | README、两批日志、报告、后改源码并存 | 人工版需补 commit/hash 绑定 |
| 覆盖范围 | 任务专用 CIFAR-100 路径 | 同样收窄到 CIFAR-100，但保留模块化工程结构 | 都不是原仓库所有数据集的完整迁移 |
| 可维护性 | 单文件较集中，但 31 KB 文件耦合较高 | 模块职责更接近原项目，改动分散 | 人工版更利于长期开发；Codex 更利于审计与复现 |
| 性能证据 | V100 上完整冻结运行，单 seed | `sm_89` 上 seeds 1/2/3 | 当前不可做公平速度或准确率排名 |

## 7. Agent 与人工迁移的核心差距

### 7.1 目标函数不同

Codex 的隐含目标函数是“在固定输入、固定资源和统一 evaluator 下通过”；人工迁移的隐含目标函数是“把原有 PyTorch 工程解释性地改成 Jittor 并展示可训练”。这导致：

- Codex 会优先消除所有会影响 evaluator 的细小协议差异，例如 AutoAugment 随机数顺序和浮点汇总边界；
- 人工迁移愿意接受合理替代，例如 AutoAugment → ColorJitter，并在报告中解释结果差异；
- Codex 更容易形成 benchmark 专用代码；人工迁移更容易形成教学性、模块化代码。

### 7.2 Agent 更擅长“协议封闭”，人工更擅长“知识显化”

Codex 会话完整记录了哈希不一致时停止、环境探测、补丁、训练、失败和修复，适合统计修复轮数、工具调用和机器耗时。人工报告则清楚解释了 `execute`、optimizer、in-place、timm 替代、权重插值和数值稳定性等迁移知识，更适合让后来者理解“为什么这样改”。

因此，若研究问题是“谁更容易一次通过冻结 evaluator”，当前证据偏向 Codex；若研究问题是“谁产出的迁移知识更容易被开发者复用”，人工报告有明显优势。两者不能被压缩为单一准确率排名。

### 7.3 Agent 的成功也暴露了 benchmark 适配风险

Codex 第 1 轮修复并未改善模型，而是把一个矩阵元素选择为相邻 float32 表示，使 evaluator 的均值容差通过。这不是复制参考结果，也没有超出逐任务容差，但说明 agent 能够针对评测器边界做高度定向的适配。研究中应把以下两类修复分开统计：

- **语义修复：** 模型、数据、优化器、调度器、权重和预测逻辑的修复；
- **评测适配：** 序列化、dtype、舍入、文件格式和阈值边界的修复。

否则“Rounds=1”会掩盖这一轮修复的性质。

### 7.4 人工迁移的主要短板是可追溯性，而非一定是技术质量

人工版本有三随机种子实验和图表，这是 Codex 单 seed 冻结结果没有覆盖的统计信息。但当前目录同时存在旧结果、新结果、过期 README 和日志后继续修改的源码，导致第三方无法从目录本身判断哪个 commit 生成了哪个表。问题是实验工程管理，不应直接解释为迁移实现较差。

## 8. 对结论有效性的威胁

1. **DSH 证据缺失。** 这是目前最主要的阻塞，三方比较尚未成立。
2. **实验协议不相同。** Codex 与人工版本的 GPU、Jittor 版本、随机种子、数据增强和可能的模型细节不同。
3. **只有 APT/CIFAR-100 一个 case。** 结论不能外推到检测、光流或生成模型。
4. **人工过程没有会话日志。** 无法公平比较提示次数、返工轮数、人工分钟和尝试失败数。
5. **人工产物血缘不完整。** 当前代码与两批日志之间缺少 commit/hash 锁定。
6. **Codex evaluator 存在 float32 边界问题。** 一次 ULP 级修复可改变 pass/fail，应在后续实验统一处理。
7. **墙钟时间不等于劳动时间。** Agent 等待 GPU 与人工离线思考都不能从现有记录中准确分离。

## 9. 下一轮公平实验设计

为了真正研究 agent 迁移和人工迁移的差距，建议把下一轮设计为同一冻结协议下的三组盲测。

### 9.1 必须统一的变量

- 同一 APT 原始 commit/source-tree hash；
- 同一 ViT-B/16 checkpoint 及文件 hash；
- 同一 CIFAR-100 数据快照、任务顺序和 seed；
- 同一 V100、CUDA、Python、Jittor 版本；
- 同一 AutoAugment 实现与 RNG 规则；
- 同一最大修复轮数，例如 8；
- 同一 evaluator，且运行者不能修改 evaluator；
- 每轮保存仓库树哈希、完整日志、命令、环境和结果哈希。

人工组也必须从完全相同的原始树开始，不能直接使用 `D:\大学\新芽\APT-Jittor` 的已有迁移结果作为起点；它可以作为经验参考，但不能作为盲测输入。

### 9.2 建议分成两层指标

**第一层：冻结任务结果**

- RN：完整训练/推理能否运行；
- Numerical/Task Alignment：准确率矩阵、最终预测、遗忘率是否在容差内；
- Final Accuracy、Forgetting；
- Repair Rounds；
- GPU hours、总墙钟时间。

**第二层：迁移过程与工程质量**

- 首次可运行时间、首次完整训练通过时间；
- 语义修复轮数与评测适配轮数；
- 修改文件数、增删行、原源码侵入度；
- 禁止依赖/框架残留；
- 断点恢复能力；
- 源码—会话—结果的哈希可追溯性；
- 模块可维护性与原项目功能覆盖率；
- 人工干预分钟数和每轮指令字数。

### 9.3 DSH 补齐清单

V100 恢复后，应优先导出而不是直接重跑覆盖现场：

1. DSH APT 原始会话（含完整工具调用和补丁）；
2. `/root/data-fs/APT/workspaces/dsh/repo`；
3. `/root/data-fs/APT/results/dsh`；
4. round 日志、训练日志和 evaluator JSON；
5. 起始与结束 source-tree hash；
6. 若不存在这些目录，则正式记为“DSH APT 尚未执行”，再从冻结源树启动新实验。

### 9.4 evaluator 修订建议

下一轮开始前应做一次版本化决定：

- 推荐把准确率矩阵均值统一转为 float64 后计算，并对序列化后的可表示误差留出明确容差；
- 或者保持当前 evaluator 不动，但在协议中声明 float32/ULP 行为，并要求三组使用同一个冻结版本。

修订后必须生成新的 evaluator hash，不能只在某一组实验中临时替换。

## 10. 最终结论

在现有证据下，可以确认 **Codex 已经完成了 APT 的原生 Jittor 训练迁移，并在冻结 V100 evaluator 下通过**；其主要优势是协议忠实、产物可追踪、可恢复和结果验证细。人工迁移则完成了更传统、更模块化的框架改写，明确沉淀了 API 替换和数值稳定性经验，并提供了三 seed 结果；其当前主要问题是与 Codex 协议不一致，以及代码、日志、README 和报告之间缺少不可变版本绑定。

因此，目前最稳妥的研究表述不是“Codex 比人工迁移高 1.72 个百分点”，因为 `89.16%` 与 `87.44%` 来自不同实验条件；而是：

> Codex 在冻结、外部评测驱动的任务中表现出更强的协议闭合和审计性；人工迁移在工程结构保持与知识解释上更强。现有数据不足以比较两者的纯迁移能力或效率，更不足以加入 DSH 后做三方排序。

待 DSH APT 会话和结果补齐，并把人工版本放到同一 V100 冻结协议下重跑后，才可以形成最终三方实验报告。

## 附录 A：建议使用的数据状态编码

| 状态 | 含义 |
|---|---|
| PASS | 运行完成且通过 evaluator |
| FAIL | 已运行并有可归因的失败结果 |
| NOT_RUN | 有证据表明实验尚未启动 |
| MISSING | 应有记录但当前无法找到 |
| UNAVAILABLE | 记录可能存在，但存储/主机当前不可访问 |

当前编码：Codex=`PASS`；DSH=`MISSING + UNAVAILABLE`；人工迁移=`COMPLETED_OUTSIDE_FROZEN_PROTOCOL`。

## 附录 B：主要证据路径

```text
D:\科研\TorchPort\apt\experiment.json
C:\Users\31561\.codex\sessions\2026\08\23\rollout-2026-08-23T23-42-12-01a02f49-5f69-7ce0-ad38-e80de3f70126.jsonl
C:\Users\31561\.dsh\storages\session_projcache.json
C:\Users\31561\.dsh\sessions\--D-~79D1~7814-TorchPort--\*.jsonl.zstd

D:\大学\新芽\apt
D:\大学\新芽\APT-Jittor
D:\大学\新芽\任务三-代码迁移报告.docx
D:\大学\新芽\任务四-基于 Jittor 框架的 APT 持续学习模型迁移与验证.docx
D:\大学\新芽\任务四-基于 Jittor 框架的 APT 持续学习模型迁移与验证.pdf
D:\大学\新芽\comparison_results\alignment_report.md
D:\大学\新芽\comparison_results\alignment_summary.json
D:\大学\新芽\seed1.log
D:\大学\新芽\seed2.log
D:\大学\新芽\seed3.log
D:\大学\新芽\APT-Jittor\seed1.log
D:\大学\新芽\APT-Jittor\seed2.log
D:\大学\新芽\APT-Jittor\seed3.log
```
