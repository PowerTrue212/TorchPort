# CLIP 训练检验准备（2026-10-28）

状态：**已完成源端 oracle 和控制端目标参考试跑，不是冻结的训练 GATE，也没有启动 CLIP 训练 Agent trial。**
已有的 full-r3 只通过固定推理验收；训练不能据此判通过。

## 已固定的试跑材料

- 沿用推理 pilot 的官方 ViT-B/32 checkpoint（302 项）和已预处理的 CIFAR-10 图像。
- `prepare_train_inputs.py` 从每类 10 张固定图像中，按原顺序取前 4 张：每步每类 1 张，
  配各类不同的固定文本，共 4 步、每步 10 对。生成的原始 NPY 和输入哈希清单在
  `migrations/CLIP/data/train-pilot/`。该目录不进入 Git。
- `train_source_adapter.py` 使用官方 CLIP 模型、双向对比交叉熵、全参数 AdamW
  （lr=1e-5，weight_decay=0）及 4 步线性衰减。这是可行性配置；进入任务前仍须
  固定训练预算、优化器语义和负对照。
- `probe_train_source.py` 在 V100 上加载 302/302 项权重并执行上述 4 步。源代码与
  既有远端推理 pilot 的官方副本逐文件哈希核对后，复制到
  `/root/data-fs/torchport-clip-train-pilot/`；输入和脚本上传后也核对哈希。
- `prepare_train_task.py` 生成独立的 4 步 `train-step` pilot TaskSpec；
  `preflight_train_task.py` 通过真实远端 `verify` 路径建立源 oracle。两者只在
  控制端运行，TaskSpec 保存在 `migrations/CLIP/outputs/train-pilot/`。

## V100 实测

PyTorch 2.4.1+cu121、V100-SXM2-16GB、fp32，开启确定性算法并设置
`CUBLAS_WORKSPACE_CONFIG=:4096:8`，关闭 TF32。两次独立进程的 4 个 loss 和
4 个学习率逐项完全相同：

| 步 | loss | 更新前学习率 |
| ---: | ---: | ---: |
| 0 | 0.2787350118160248 | 1e-5 |
| 1 | 0.514252781867981 | 7.5e-6 |
| 2 | 0.27179014682769775 | 5e-6 |
| 3 | 0.6177530884742737 | 2.5e-6 |

峰值 CUDA 已分配内存 2,761,024,512 字节（约 2.57 GiB）。两次原始 JSON 在
`migrations/CLIP/outputs/train-pilot/source-probe*.json`，远端原件仍保留。
首次运行因未设置 `CUBLAS_WORKSPACE_CONFIG` 在确定性矩阵乘处中止；补齐该
环境变量后两次完整通过。失败运行没有作为源端重复性证据。

首次建 oracle 的 `clip-train-source-r0` 在 FP64 标定路径失败：官方 CLIP 的
`LayerNorm` 为兼容 fp16 会强制把输入转成 FP32，但整网 `.double()` 后权重
为 FP64。受控训练适配层现在只对 FP64 标定分支保留 FP64 LayerNorm；FP32
分支仍走原来的 FP32 计算。改动后复跑 4 步，loss 和学习率与改动前逐项一致。
旧的 r0 失败报告及远端原始产物保留，未改写。

`clip-train-source-r1` 使用新 TaskSpec 建成完整 source oracle：两次源端运行，
302 项权重全部加载，`train_fp64.npz` 与 `train_alt.npz` 均生成，oracle 的
`ok=true`。oracle digest 为
`248146ef5d44ce1693c0d19a7c958f88faa18f5aaae3740fa9c46d151a723ac3`。
未实现训练入口的 target seed 按预期返回 `RUN_FAILED`；这不表示正确 Jittor
目标端已校准。`oracle.json` 已复制到
`migrations/CLIP/outputs/train-pilot/oracle-r1.json`。五个原始 NPZ 已复制到
`D:\科研\CLIP-Benchmark-v1\controller\clip-train-source-r1\oracle-raw\`，
逐文件 SHA-256 均与 `oracle.json` 中记录相同；远端原件也保留。

## 控制端 Jittor 参考试跑

从 `clip-full-r3` 的**冻结推理交付快照**另复制出控制端参考仓库；原 trial
与 Agent 工作区均未改动。`train_target_reference.py` 复用其已通过推理检查的
双塔算子，但训练入口绕开 `run.py` 中的 `jt.no_grad()`，加上双向对比损失和
AdamW 更新。这只是待验候选，不能预称正确参考。

- `clip-train-reference-r0` 在构建优化器时失败：原推理加载器调用 Jittor
  `eval()`，使参数停止求导。已保留 r0 适配代码与报告。
- r1 在权重加载后恢复训练和参数梯度。目标端可运行、结构检查通过，但
  隐藏 `verify` 返回 `PARITY_FAILED`：首个超限为第 0 步某个 attention
  参数更新的统计量。重复执行仍失败，首个超限参数有所变化。按当前
  `k=8` 噪声包络，本地重算分别有 30 和 22 个张量超限；这不能靠总输出
  误差小来改判。
- 两次目标端原始输出比较：2422 个张量中 1934 个逐位不同，合计
  5,340,748 个元素不同，最大绝对差约 `1.30e-4`。同一 source oracle 的
  两次 FP32 输出在 2422 个张量上逐位一致。目标端波动的算子级根因尚未确认。

r1 两次 `verify-result.json` 与各自的目标输出 NPZ、capture、manifest 位于
`D:\科研\CLIP-Benchmark-v1\controller\clip-train-reference-r1\` 和
`clip-train-reference-r1-repeat\`。`compare_train_repeats.py` 生成的源端与
目标端重复性 JSON 也在对应控制目录。原始远端证据继续保留。

为拆开训练更新与前向波动，`probe_target_forward.py` 在 V100 上仅执行第 0 批的
双塔编码、特征归一化、logits 和 loss，不作参数更新。两次独立进程均加载
302 项权重；图像特征的 5120 个元素中 5014 个逐位不同（最大绝对差
`3.58e-7`），文本特征 4954 个不同（`1.79e-7`），logits 90/100 个不同
（`1.53e-5`），loss 差 `2.38e-7`。因此重复波动已经发生在参数更新之前，
而且两个编码塔都受影响；这还不能确定首个异常算子或训练超限的全部原因。
原始输出保存在 `clip-train-reference-r1/forward-probe-{a,b}.npz`，远端原件也保留。
关闭 Jittor 融合的同一探针在 V100 16 GiB 上分配约 1.18 GB 的中间张量时
显存不足，没有得到可比的无融合输出，不据此判断融合是否造成波动。

分段探针随后在每个编码块后保存原始 NPZ。原参考两次运行的图像卷积和位置
嵌入逐位一致，首次差异在 `visual.ln_pre`；其中 500 个均值中 302 个不同，
方差中 327 个不同。文本嵌入和位置嵌入逐位一致，第 0 个 Transformer 块的
`ln_1` 已不同。在相同输入上，把均值与方差的归约暂用矩阵乘法实现，两次
结果逐位一致。分段原始文件 `stages{,2,3,4}-{a,b}.npz` 在 r1 控制目录。
这将差异定位到 Jittor 归约路径，但还未做内核级根因确认。

为检验这条线索，另建控制端候选 `train_target_reference_matmul_ln.py`：仅把
LayerNorm 的均值/方差改为矩阵乘法归约，权重、输入与训练任务不变。对应的
独立前向分段探针两次运行 33 个主要阶段均逐位一致，原始文件在
`clip-train-reference-matmul-r2b/stages-{a,b}.npz`。但完整训练仍未通过：

| 控制端运行 | 隐藏 verify | 噪声包络超限张量 | 固定容差超限元素 |
| --- | --- | ---: | ---: |
| matmul-r2b | PARITY_FAILED | 25/2420 | 0/13530680 |
| matmul-r2-repeat | RUN_FAILED：Jittor `malloc(): memory corruption` | 未比较 | 未比较 |
| matmul-r2c | PARITY_FAILED | 28/2420 | 3/13530680 |
| matmul-cublas-r3 | PARITY_FAILED | 19/2420 | 0/13530680 |
| matmul-cublas-r3-repeat | PARITY_FAILED | 21/2420 | 0/13530680 |

r2b 与 r2c 的第 0 步 loss 完全相同，但首步后的 302 个参数更新统计量有
162 个不同，全部输出有 1873/2422 个张量不同；说明训练输出仍有波动。
r3 在目标环境加上 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，第 0 步
loss 仍完全相同，但首步更新统计量 160/302 个不同，全部输出
1855/2422 个张量不同。该设置未消除训练波动。r2 的一次内存损坏失败
独立保留，尚未判定是否
可复现。上述每次 `verify-result.json`、目标端输出 NPZ、capture、manifest、
job 和 stderr 均在对应的 `D:\科研\CLIP-Benchmark-v1\controller\clip-train-reference-*`
目录；源 oracle 与先前失败的 trial 均未改写。

**分段探针的限制：**它在每层后立即调用 `.numpy()`，会改变 Jittor 的执行图。
随后对 r3 候选运行不逐层同步的完整第 0 批前向，两次归一化图像特征有
950/5120 个元素不同、文本特征有 1024/5120 个不同、logits 有 27/100 个
不同，loss 差 `8.94e-8`。所以分段前向逐位一致不能证明完整前向确定；
仅凭训练更新差异也不能把剩余波动单独归因于反向传播。两次原始
`forward-{a,b}.npz` 位于 r3 控制目录的 `grad-probe/`。

同目录还保存两对第一步梯度探针：`grad-{a,b}` 在反向前同步 loss，
`grad2-{a,b}` 在反向后同步 loss。两种执行次序中，302/302 个参数梯度的
SHA-256 均不同，且 loss 自身也有细微差异（第二种为 `5.96e-8`）。这些
JSON 记录每个梯度的哈希和统计量，NPZ 保存四个选定参数的原始梯度。
这证明完整计算路径存在广泛波动，尚不能分离前向误差传播与反向内核波动。

**源端 alt 不稳定，跨控制目录不能直接比较超限数量。**核对原始 SHA-256
发现 r2 与 r3 虽然 oracle digest 同为 `248146...`，其 `outputs.npz` 和
`train_fp64.npz` 分别逐文件相同，但 `train_alt.npz` 不同：r2 为
`308151262c180767d0217de65b51d0f4d7b512b1dfc0ea260622fddf8857376c`，
r3 为 `b2c5f4ae8ca6e29e9048b52e213b32e2b86f08130b2c639fceb8b6eec523dc30`。
两份 alt 的 2420 个张量中 1501 个逐位不同，最大绝对差 `3.48e-5`。
`calibrate_steps` 的 alt 路径明确关闭确定性算法并启用 cuDNN benchmark，
因此单次 alt 不是固定基线；具体波动算子未定位。r3 版本的原始 alt 与
`oracle.json` 已复制到 r3 控制目录 `oracle-raw/`，r2 版本保留在 source-r1。

把**同一目标输出**分别按两份 alt 重算，r2b 超限数为 25/22，r2c 为
28/22，r3 为 20/19，r3-repeat 为 24/21（斜线前后分别是 r2/r3 alt）。
完整派生结果在 `oracle-raw/alt-sensitivity.json`。这证明超限数受到 alt
抽样影响；先前不能据 r2 与 r3 各自的原始超限数直接评估 cuBLAS 设置。
同一 r3 oracle 下的两次目标验证都失败，结论不受该跨目录比较问题影响。
仅作敏感性诊断时，用两份 alt 的逐张量最大噪声共同扩展包络，r2b 仍有
22 个张量超限，r3 两次仍分别有 19、21 个超限；原始派生文件为对应控制
目录的 `two-alt-diagnostic.json`。这不改写任何一次正式 `verify`，也说明
只增加这一份 alt 样本不足以让当前参考通过。
正式训练 GATE 前须冻结具体 alt 文件与哈希，或明确多次 alt 的聚合规则，
再进行同协议校准。`oracle.json` 已各自记录 alt 文件哈希，原始溯源完整；
只按相同 task digest 合并实验会遗漏这项差别。

### 固定参考重复性包络的离线敏感性检查（2026-10-29）

为检查 SimCSE 提出的 pilot 规则是否也能解释 CLIP 参考波动，
`analyze_repeat_envelope.py` 以 r1 与 r1-repeat 的目标输出逐张量距离
建立**候选无关**的固定参考基数，和 source FP32/FP64/alt 距离取最大值，
去掉 `ustat` 的 sum，只对其 L2/max_abs 和其他全部输出按
`k=8`、`floor=1e-7` 比较。只复判已有原始 NPZ，未调用 GPU，
未改写原隐藏验收。

这次诊断明确选定 r3 的 `train_alt.npz`，SHA256 为
`b2c5f4ae8ca6e29e9048b52e213b32e2b86f08130b2c639fceb8b6eec523dc30`；
它与 r2 的 alt（`308151262c180767d0217de65b51d0f4d7b512b1dfc0ea260622fddf8857376c`）
分开留存，不再按同一 task digest 隐式混用。r1 两次用于标定；
r2b、r2c、r3、r3-repeat 是留出诊断。六份输出的 2,420 个 `output.*`
张量均无超限，最大阈值使用率依次为 0.372、0.414、0.528、
0.566、0.568、0.571。改用 r2 alt 后，这六个数完全相同，
说明这组数据的固定参考重复性基数支配了 alt 差异。
完整结果和文件哈希在控制端 `clip-train-source-r1/oracle-raw/`
的 `fixed-r1-peer-alt-r3-diagnostic.json` 与
`fixed-r1-peer-alt-r2-diagnostic.json`。

这是**pilot 敏感性分析**，不能据此认定 r2/r3 的 Jittor 实现正确。
r1 两次参考尚未通过独立规则证明正确，CLIP 负对照未在该规则下验证，
且只有两次标定重复。正式 GATE 的 alt 仍须按新协议冻结并校准；
上述 r3 哈希只固定本次离线诊断。

远端 `/root/data-fs` 最新约 137 GB 可用；根分区显示 0 可用。后续训练任务的
同步目录、HOME、Jittor 缓存和临时文件应放在数据盘，不继续写根分区。

## 尚缺的验收准备

1. 调查 Jittor 训练重复波动与超限统计量，校准一个能稳定通过的正确目标
   实现；检查全参数更新 sketch、loss、学习率和 checkpoint 回环，并以
   源端负例校准误判与漏检。当前参考候选尚未通过。
2. 正式 GATE 的步数、batch、容差和 GPU 资源预算仍须在 pilot 后确定。
3. 从同一干净 seed 建立 none/full 训练工作区及同一提示词，证明容器隔离，
   再启动 Agent trial。保留推理 trial 的原始结果，不覆盖。

本记录证明**源端四步训练在 V100 上可运行且重复稳定**，并发现目标端
归约和训练更新路径的跨进程波动；尚不能证明 Jittor 迁移正确或 Benchmark v1
的正式结果。
