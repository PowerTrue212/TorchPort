# Held-out COD 迁移候选调研

更新时间：2026-10-08

范围：`ChunmingHe/RUN`、`Sssssuperior/VSCode` 的 VSCode-v2、`yjybuaa/PlantCamo`、`zhengpeng7/birefnet`、`ai4colonoscopy/PraNet-V2`

## 调研口径

- 本文只读取 GitHub 公共 API、仓库文件和仓库链接的公开网页；没有下载权重或数据集，也没有访问远端 GPU。
- commit、tag、代码路径、README 声明和 GitHub release 文件大小属于已核实的仓库证据。所有运行可行性、显存和 trace 体量判断都写成“估计（未实测）”。
- “可非交互下载”只表示仓库给出了可以直接取文件的 URL，不表示本次已经下载或校验哈希。Google Drive/Baidu 的 `view`、文件夹或提取码页面不算非交互式直链。
- “未发现 Jittor 端口”只说明官方仓库树和按项目精确名称进行的公开网页检索没有命中；仓外是否存在未索引、私有或改名端口仍未核实。

## 总览

| 候选 | 冻结版本 | 官方输入 | 现有四套 COD 数据适配 | 权重自动获取 | Jittor 状态 | 建议 |
| --- | --- | ---: | --- | --- | --- | --- |
| BiRefNet-COD | main `ebcc0bc`；最新 tag `v1` | 1024 | 四套齐全，但目录名和子目录要适配 | GitHub release 直链 | 未发现官方端口 | 主候选 1 |
| VSCode-v2-T stage2 | main `d0ec32e`；无 tag | 352 | COD10K/CAMO/NC4K，缺 CHAMELEON；`RGB/GT` | Baidu 提取码页 | 未发现官方端口 | 主候选 2 |
| PlantCamo | main `867452b`；无 tag | 704 | 不适配；只测 PlantCAMO1250，`rgb/gt` | Google Drive/Baidu 页面 | 未发现官方端口 | 主候选 3 |
| RUN | main `878208d`；无 tag | 384 | 四套齐全，原生 `Imgs/GT` | Google Drive 文件夹页 | 未发现官方端口 | 高难候选 |
| PraNet-V2 | main `ca08522`；无 tag | 352 | 不适配；四套息肉数据，`images/masks` | Google Drive 页面 | 官方仓库已有 Jittor 推理 | 对照 |

## 1. BiRefNet-COD — 主候选 1

- **版本。** 最新 main commit 是 [`ebcc0bc8ec7fe919cec829f2dea656b3078acddc`](https://github.com/ZhengPeng7/BiRefNet/commit/ebcc0bc8ec7fe919cec829f2dea656b3078acddc)，提交日期为 2026-09-02；最新 tag/release 是 [`v1`](https://github.com/ZhengPeng7/BiRefNet/releases/tag/v1)，tag 指向 `a0cf9925880620000aa2d1948d61bf659ddfdfaa`。main 在持续演进，正式任务必须冻结 commit，不能只写 `main`。
- **模型与预训练依赖。** 原论文 COD 模型使用 Swin-V1-Large；[模型表](https://github.com/ZhengPeng7/BiRefNet/blob/ebcc0bc8ec7fe919cec829f2dea656b3078acddc/README.md#model-zoo)写明训练集为 COD10K-TR + CAMO-TR。[当前配置](https://github.com/ZhengPeng7/BiRefNet/blob/ebcc0bc8ec7fe919cec829f2dea656b3078acddc/config.py)默认也是 `swin_v1_l`，但默认 task 是 DIS5K，运行 COD 前必须显式冻结 COD 配置。完整 COD checkpoint 推理时用 `bb_pretrained=False`，不应再单独联网取 backbone。
- **输入分辨率。** 当前固定 `config.size=(1024, 1024)`；`inference.py` 也默认采用该配置，可选择原分辨率或显式分辨率。证据见 [`config.py`](https://github.com/ZhengPeng7/BiRefNet/blob/ebcc0bc8ec7fe919cec829f2dea656b3078acddc/config.py)和 [`inference.py`](https://github.com/ZhengPeng7/BiRefNet/blob/ebcc0bc8ec7fe919cec829f2dea656b3078acddc/inference.py)。
- **官方训练权重。** 推荐冻结 [`BiRefNet-COD-epoch_125.pth`](https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-COD-epoch_125.pth)：GitHub release，885,082,437 bytes（约 844 MiB），可非交互下载。相同 release 还有 COD ONNX，972,666,916 bytes；TorchPort 迁移任务应使用 `.pth`，不要把 ONNX 推理当作 PyTorch→Jittor 迁移。README 另给 [Google Drive 镜像](https://drive.google.com/file/d/1tM5M72k7a8aKF-dYy-QXaqvfEhbFaWkC/view)，其大小和非交互下载能力未核实。
- **测试数据与目录。** COD 配置列出 CHAMELEON、NC4K、TE-CAMO、TE-COD10K；[`dataset.py`](https://github.com/ZhengPeng7/BiRefNet/blob/ebcc0bc8ec7fe919cec829f2dea656b3078acddc/dataset.py)读取 `<data_root>/COD/<dataset>/im` 与 `gt`。数据集覆盖与现有四套一致，但不直接匹配 `TestDataset/<name>/Imgs` 与 `GT`：需要只做路径映射/软链接的适配层，并冻结 `TE-CAMO→CAMO`、`TE-COD10K→COD10K` 的命名映射。
- **评测脚本与指标。** [`eval_existingOnes.py`](https://github.com/ZhengPeng7/BiRefNet/blob/ebcc0bc8ec7fe919cec829f2dea656b3078acddc/eval_existingOnes.py)调用仓库内 [`evaluation/metrics.py`](https://github.com/ZhengPeng7/BiRefNet/blob/ebcc0bc8ec7fe919cec829f2dea656b3078acddc/evaluation/metrics.py)。脚本默认只算 S-measure 和 MAE；`--metrics all` 还支持 E/F/weighted-F、MSE、HCE、mBA、boundary IoU。正式任务要预先冻结指标子集，避免默认值与论文表格字段不一致。
- **Python 依赖。** [`requirements.txt`](https://github.com/ZhengPeng7/BiRefNet/blob/ebcc0bc8ec7fe919cec829f2dea656b3078acddc/requirements.txt)要求 Torch≥2.5、torchvision、timm、kornia、einops、accelerate 等。当前默认 `dec_att=ASPPDeformable`，最终落到 `torchvision.ops.deform_conv2d`；同时启用 Swin SDPA、`torch.compile` 和 bf16。V100 没有原生 bf16，因此正式 V100 协议应冻结为 fp16 或 fp32，并明确是否关闭 compile；这样做是新运行配置，不得和默认结果混写。相关证据见 [`config.py`](https://github.com/ZhengPeng7/BiRefNet/blob/ebcc0bc8ec7fe919cec829f2dea656b3078acddc/config.py)和 [`deform_conv.py`](https://github.com/ZhengPeng7/BiRefNet/blob/ebcc0bc8ec7fe919cec829f2dea656b3078acddc/models/modules/deform_conv.py)。
- **现有 Jittor 端口。** 官方仓库没有 Jittor 目录或说明；仓外精确名称检索未发现端口，但这个否定结论未核实为穷尽结果。
- **V100-16G 与 trace 风险。** 可行性估计（未实测）：README 声称标准模型在 RTX 4090、fp16、1024² 输入下占 3.45 GB，因此 V100-16G 做 batch=1 推理大概率有显存余量；V100 上 SDPA、deform conv 和 fp16 路径是否可用仍需 smoke。trace 风险为**很高**（未实测）：1024²、Swin-L、多尺度双输入与 deformable ASPP 会产生大张量；首轮应限制到 backbone stage、decoder block 和最终输出，不应无差别保存全部中间值。
- **迁移难点。** 主要是 deformable convolution、einops 重排、Swin attention/SDPA、配置文件在 import 时决定网络结构，以及 release 权重与所选 commit/config 的严格匹配。优点是权重有 GitHub 直链、四套 COD 数据齐全，是五个候选里最容易做成可自动复验任务的一个。

## 2. VSCode-v2-T stage2 — 主候选 2

- **版本。** 最新 main commit 是 [`d0ec32e2f1b2dfd4153afcf2d807476fe2c0ead4`](https://github.com/Sssssuperior/VSCode/commit/d0ec32e2f1b2dfd4153afcf2d807476fe2c0ead4)，提交日期为 2025-12-08；[tags 页面](https://github.com/Sssssuperior/VSCode/tags)和 API 均无 tag。
- **模型与预训练依赖。** VSCode-v2 是多任务模型，提供 Swin-T/S 两个 backbone 和两阶段训练。主候选建议选最终的 `VSCode-v2-T(stage2)`，README 报告 69.8M 参数；[`train_test_eval.py`](https://github.com/Sssssuperior/VSCode/blob/d0ec32e2f1b2dfd4153afcf2d807476fe2c0ead4/VSCode2/train_test_eval.py)默认 backbone 预训练文件为 `swin_tiny_patch4_window7_224.pth`。正式推理只应加载完整 stage2 checkpoint，是否还会触发 backbone 文件读取需在本地 smoke 中核实。
- **输入分辨率。** `img_size` 默认 352；[`Testing.py`](https://github.com/Sssssuperior/VSCode/blob/d0ec32e2f1b2dfd4153afcf2d807476fe2c0ead4/VSCode2/Testing.py)和 decoder 的 Fold/reshape 都假定固定尺寸。
- **官方训练权重。** 最终 T stage2 在 [Baidu](https://pan.baidu.com/s/1U5vrfw9CFwQevHuOaEA8cg?pwd=8imx)，提取码 `8imx`；S stage2 在 [Baidu](https://pan.baidu.com/s/1T6l3UL7yQvtnDRs40quKwQ?pwd=4r7b)，提取码 `4r7b`。两者均是 Baidu 网盘页面，仓库没有给 v2 的 GitHub release/Hugging Face/非交互直链，文件大小未公布。旧 VSCode v1 有 Google Drive 镜像，但不能替代 v2 权重。
- **测试数据与目录。** [`train_test_eval.py`](https://github.com/Sssssuperior/VSCode/blob/d0ec32e2f1b2dfd4153afcf2d807476fe2c0ead4/VSCode2/train_test_eval.py)的 RGB COD 测试集是 COD10K、CAMO、NC4K，没有 CHAMELEON。[`dataset.py`](https://github.com/Sssssuperior/VSCode/blob/d0ec32e2f1b2dfd4153afcf2d807476fe2c0ead4/VSCode2/dataset.py)和 README 使用 `Data/COD/rgb/Test/<dataset>/RGB`、`GT`，不直接匹配 `Imgs/GT`。需要目录适配；是否把 CHAMELEON 加入官方模型评测尚未核实，不能默认补入正式协议。
- **评测脚本与指标。** `VSCode2/Evaluation/main.py` 和 [`evaluator.py`](https://github.com/Sssssuperior/VSCode/blob/d0ec32e2f1b2dfd4153afcf2d807476fe2c0ead4/VSCode2/Evaluation/evaluator.py)计算 MAE、max/mean F-measure、max/mean E-measure、S-measure、AP 和 AUC。评测默认在 CUDA 上运行，需与模型推理成本分开记录。
- **Python 依赖。** README 只写 PyTorch≥1.6、torchvision≥0.7，没有锁定 requirements。代码实际还导入 timm、ptflops、fvcore、OpenCV、Pillow 和 NumPy；RGB COD 路径未发现自定义 CUDA op。Flownet2 只服务视频任务，不应进入本次 RGB COD 环境。
- **现有 Jittor 端口。** 官方仓库没有 Jittor 路径；仓外精确名称检索未发现端口，但是否存在改名端口未核实。
- **V100-16G 与 trace 风险。** 可行性估计（未实测）：Swin-T、batch=1、352² 在 V100-16G 上大概率可推理，但 69.8M 参数只是 README 数值，不等于显存实测。trace 风险为**高**（未实测）：多任务 prompt、MoPE、共享/专用双分支和四尺度 mask/contour 输出会带来大量调用与张量；应先冻结为 RGB COD 单任务并只捕获该路径。
- **迁移难点。** 标准算子为主，但工程复杂度高：训练/测试目录重复、checkpoint 路径硬编码、state_dict 还要手工去前缀，任务由字符串分支控制。最大阻塞是 v2 权重只有需要提取码的 Baidu 页面；在移入隐藏、可校验的本地 artifact 前不宜进入正式 replicate。

## 3. PlantCamo — 主候选 3

- **版本。** 最新 main commit 是 [`867452b7b43cd0425e8d988203930065e7fed6c2`](https://github.com/yjybuaa/PlantCamo/commit/867452b7b43cd0425e8d988203930065e7fed6c2)，提交日期为 2024-12-11；[tags 页面](https://github.com/yjybuaa/PlantCamo/tags)和 API 均无 tag。
- **模型与预训练依赖。** [`backbone/PLCamo.py`](https://github.com/yjybuaa/PlantCamo/blob/867452b7b43cd0425e8d988203930065e7fed6c2/backbone/PLCamo.py)使用仓库内 PVTv2-B2 和一个迭代式 decoder，初始化时固定读取 `./pretrained_pvt/pvt_v2_b2.pth`。因此完整模型 checkpoint 之外还需 ImageNet PVTv2-B2 预训练文件，且两者都要进入候选身份/哈希清单。
- **输入分辨率。** [`MyTesting.py`](https://github.com/yjybuaa/PlantCamo/blob/867452b7b43cd0425e8d988203930065e7fed6c2/MyTesting.py)默认 `testsize=704`；模型注释中仍有 448 的旧说明，正式协议应以实际测试参数 704 为准。
- **官方训练权重。** 最终 `Net_epoch_best.pth` 提供 [Google Drive](https://drive.google.com/file/d/1-GP18g79xszyAnvZ1ZzTXzhFoH7gzYed/view?usp=sharing) 和 [Baidu](https://pan.baidu.com/s/1eZpqxx5b5Y_V9klLQxf5WQ)（提取码 `b98f`）；PVTv2-B2 预训练权重也提供 [Google Drive](https://drive.google.com/file/d/1-AFs2dP3p0OMw3Vbnf0tMsgkYbpT6C23/view?usp=sharing) 和 Baidu。都是 `view`/提取码页面，没有仓库发布的非交互直链；大小未公布。
- **测试数据与目录。** 官方脚本只跑 PlantCAMO1250，读取 `datasets/PlantCAMO1250/test/rgb` 与 `gt`。它不匹配 COD10K/CAMO/NC4K/CHAMELEON 的 `TestDataset/<name>/Imgs/GT`，也没有用这四套数据做官方测试。代码中的四套数据循环只是注释；用官方 PlantCamo 权重在四套通用 COD 数据上的效果未核实。
- **评测脚本与指标。** [`metric_caller.py`](https://github.com/yjybuaa/PlantCamo/blob/867452b7b43cd0425e8d988203930065e7fed6c2/metric_caller.py)基于 PySODMetrics，报告 S-measure、weighted F、MAE，以及 adaptive/mean/max E-measure 和 F-measure。
- **Python 依赖。** 仓库没有 requirements/environment 文件。可从代码核实的直接依赖包括 torch、torchvision、timm、NumPy、SciPy、OpenCV、Pillow 和 PySODMetrics；未发现 mmcv、自定义 CUDA op 或 deformable conv。
- **现有 Jittor 端口。** 官方仓库没有 Jittor 路径；仓外精确名称检索未发现端口，但否定结论未核实为穷尽结果。
- **V100-16G 与 trace 风险。** 可行性估计（未实测）：官方只声明在 24 GB RTX 3090 上做训练/测试；batch=1 的 704² 推理在 V100-16G 上可能可行，但必须用真实权重 smoke，不能从训练卡显存反推成功。trace 风险为**中高**（未实测）：PVT token 和 704² 早期 feature 很大，但网络算子常规，按 stage 捕获应可控。
- **迁移难点。** 算子迁移相对直接，主要是 PVT attention、LayerNorm、插值、in-place 累加和两轮反馈。真正的实验阻塞是它使用独立 PlantCAMO 数据和私有网盘式权重分发；若纳入主任务，需要把“不同测试任务”明确写入分析，不能与四套标准 COD 的成功率直接横比。

## 4. RUN — 高难候选

- **版本。** 最新 main commit 是 [`878208d5154577075fe15156866970f2a57c2fb8`](https://github.com/ChunmingHe/RUN/commit/878208d5154577075fe15156866970f2a57c2fb8)，提交日期为 2025-06-03；[tags 页面](https://github.com/ChunmingHe/RUN/tags)和 API 均无 tag。
- **模型与预训练依赖。** [`lib/Network.py`](https://github.com/ChunmingHe/RUN/blob/878208d5154577075fe15156866970f2a57c2fb8/lib/Network.py)的实际 `Network` 使用 timm ResNet-50 (`pretrained=True`) backbone，结合频域/可逆模块和 Selective Mamba。完整 checkpoint 之外，timm 的 ImageNet ResNet-50 权重是否已完全包含在 checkpoint 中需要通过离线 smoke 核实；正式运行必须禁止隐式联网下载。
- **输入分辨率。** [`Test.py`](https://github.com/ChunmingHe/RUN/blob/878208d5154577075fe15156866970f2a57c2fb8/Test.py)默认 384。
- **官方训练权重。** README 的 [Models & Results](https://drive.google.com/drive/folders/1rA8RfYDmEkUESsRAEgVVqCj5ImkRNTsE?usp=sharing) 是 Google Drive 文件夹。仓库没有 release/Hugging Face/直接文件名与大小清单，也没有非交互式下载 URL；具体 checkpoint 文件、大小和哈希均未核实。
- **测试数据与目录。** `Test.py` 原生循环 COD10K、CAMO、CHAMELEON、NC4K，并读取 `<root>/<dataset>/Imgs` 和 `GT`，与现有 COD `TestDataset` 布局直接匹配。这是五个候选中唯一不需要数据目录改名的模型。
- **评测脚本与指标。** `Test.py` 在线打印 MAE；README 把正式评测指向 [CODToolbox](https://github.com/DengPingFan/CODToolbox) 的 MATLAB `main.m`。仓库也含 `util/eval_functions.py`，但没有一个已冻结、直接调用它的完整 Python 评测入口。正式任务应选定 CODToolbox commit 和指标字段；未经冻结前，除 MAE 外的确切正式指标集合未核实。
- **Python 依赖。** [`environment.yml`](https://github.com/ChunmingHe/RUN/blob/878208d5154577075fe15156866970f2a57c2fb8/environment.yml)包含 torch 2.1/CUDA 11.8、timm、einops、mamba-ssm 2.2.2 和 Triton；仓库同时有另一份旧 `requirements.txt`（torch 1.8、mmcv 1.4.2 等），两份环境互相冲突。实际 forward 从 `mamba_ssm.ops.selective_scan_interface` 导入 selective scan，属于需要编译/加载 CUDA 实现的关键依赖。
- **现有 Jittor 端口。** 官方仓库没有 Jittor 路径；仓外精确名称检索未发现 RUN 的同名端口，但检索容易被通用单词 “RUN” 污染，因此仓外否定结论尤其未核实。
- **V100-16G 与 trace 风险。** 可行性估计（未实测）：384²、batch=1 的张量规模本身可能可行，但 `mamba-ssm`/Triton 在 V100、目标 CUDA 和 torch 版本上的构建兼容性是先决阻塞；官方 `Test.py` 还按所有可见 GPU 启动 NCCL DDP，并硬编码 `CUDA_VISIBLE_DEVICES=2,3`，不能直接作为单卡 V100 evaluator。trace 风险为**很高**（未实测）：多阶段 reversible/frequency 分支、多个全分辨率输出和 selective scan 会生成大量证据。
- **迁移难点。** 这是唯一明确依赖 Mamba 自定义 CUDA selective scan 的候选。更严重的是 `Network.forward` 会现场创建 `Selectivemamba().cuda()`，而 `Selectivemamba.forward` 又现场创建两个 `VSSBlock(...).cuda()`；这些模块没有注册到父模型，参数身份与 checkpoint 覆盖范围需要先核实。它适合检验 localized 反馈面对自定义算子与动态图构造的能力，不适合第一批公平 replicate。

## 5. PraNet-V2 — Jittor 对照

- **版本。** 最新 main commit 是 [`ca08522ae77cbda6c5056609e7951b502dbb08f2`](https://github.com/ai4colonoscopy/PraNet-V2/commit/ca08522ae77cbda6c5056609e7951b502dbb08f2)，提交日期为 2026-03-21；[tags 页面](https://github.com/ai4colonoscopy/PraNet-V2/tags)和 API 均无 tag。
- **模型与预训练依赖。** 二值分割部分提供 PraNet-V2（Res2Net-50）与 PVT-PraNet-V2（PVTv2-B2）。README 要求额外准备 `res2net50_v1b_26w_4s-3cf99910.pth` 或 `pvt_v2_b2.pth`；完整模型分别是 RES-V2 与 PVT-V2。证据见 [README 的 backbones/weights 部分](https://github.com/ai4colonoscopy/PraNet-V2/blob/ca08522ae77cbda6c5056609e7951b502dbb08f2/README.md#backbones-preparation)。
- **输入分辨率。** [`binary_seg/MyTest_med.py`](https://github.com/ai4colonoscopy/PraNet-V2/blob/ca08522ae77cbda6c5056609e7951b502dbb08f2/binary_seg/MyTest_med.py)默认 352。
- **官方训练权重。** PraNet-V2 的 [RES-V2.pth](https://drive.google.com/file/d/1MFPcbD4Lt3o4d9RAwsT4wfLT32KtLqmF/view?usp=sharing)和 PVT-PraNet-V2 的 [PVT-V2.pth](https://drive.google.com/file/d/1y5i06UYyESxQfawj9swNhyQcyPX6OD1_/view?usp=sharing)均托管在 Google Drive；预训练 backbone 也在 Google Drive。仓库没有 release/HF 非交互直链，文件大小未公布。
- **测试数据与目录。** 官方二值测试集是 CVC-300、CVC-ClinicDB、Kvasir、ETIS-LaribPolypDB，读取 `binary_seg/data/TestDataset/<dataset>/images` 与 `masks`。它既不是 COD10K/CAMO/NC4K/CHAMELEON，也不匹配 `Imgs/GT`；不能把医学息肉指标与 COD 指标直接合并。
- **评测脚本与指标。** [`binary_seg/eval.py`](https://github.com/ai4colonoscopy/PraNet-V2/blob/ca08522ae77cbda6c5056609e7951b502dbb08f2/binary_seg/eval.py)默认报告 mean Dice、mean IoU、weighted F、S-measure、mean E-measure 和 MAE。
- **Python 依赖。** [`pranet2.yaml`](https://github.com/ai4colonoscopy/PraNet-V2/blob/ca08522ae77cbda6c5056609e7951b502dbb08f2/pranet2.yaml)固定 torch 2.0.1/CUDA 11.8，并含 timm、kornia、OpenCV、SciPy、MedPy、ptflops 等；二值 PyTorch 路径没有发现 mmcv 或自定义 CUDA op。
- **现有 Jittor 端口。** 已有官方 [`binary_seg/jittor`](https://github.com/ai4colonoscopy/PraNet-V2/tree/ca08522ae77cbda6c5056609e7951b502dbb08f2/binary_seg/jittor) 推理和评测代码，覆盖 PraNet-V1、PVT-PraNet-V1、PraNet-V2、PVT-PraNet-V2，并发布了 PyTorch/Jittor 指标与速度表。因此它不是 held-out 迁移任务，应只作为“已知端口能否被 TorchPort 正确验收/定位”的正对照。
- **V100-16G 与 trace 风险。** 可行性估计（未实测）：352²、batch=1 的四个模型在 V100-16G 上大概率可运行，官方 Jittor 页面还给出批量速度结果，但未说明测试 GPU，不能当作 V100 证据。trace 风险为**中**（未实测）：Res2Net/PVT 与四级 reverse-attention 输出较多，但输入和网络规模低于 RUN/BiRefNet。
- **迁移难点。** 标准卷积、PVT attention、插值和双监督 reverse attention 为主。主要价值不是迁移难度，而是已有 Jittor 实现和公开对齐表；应检查现有端口的权重加载完整性与 TorchPort candidate 身份，不能把它计入 held-out 成功率。

## 推荐排序与进入条件

### 三个主候选

1. **BiRefNet-COD。** 四套 COD 测试齐全，权重有 GitHub release 直链和确定大小，最容易建立隐藏、可复验的 artifact。进入条件：冻结 commit/config，验证 `v1` COD checkpoint 与该结构严格匹配；在 V100 上把 bf16/compile 策略写入新协议；先证明 deform conv 两端可运行。
2. **VSCode-v2-T stage2。** 输入小、RGB COD 路径以标准 PyTorch 算子为主，可代表多任务 transformer/prompt 迁移。进入条件：把 Baidu 权重合法镜像到实验环境并记录哈希；确认 checkpoint 可离线加载；冻结只含 COD-RGB 的单任务入口；决定是否排除 CHAMELEON，而不是临时补数据。
3. **PlantCamo。** 网络较独立、标准算子占主导，适合作为另一种 PVT/迭代 decoder 任务。进入条件：镜像完整模型与 PVT backbone 权重并记录哈希；在 V100-16G 做 704² smoke；接受它是 PlantCAMO 专属任务，分析时与四套通用 COD 分层报告。

### 一个高难候选

4. **RUN。** 保留为 advanced：它原生匹配四套 COD 数据，但 Mamba selective-scan CUDA、forward 内动态创建模块、相互冲突的环境清单和强制多卡 DDP 都会同时改变迁移与运行环境难度。只有在单卡离线 PyTorch oracle 可稳定复现、checkpoint 覆盖范围查清、Mamba 依赖可在 V100 环境构建后，才适合纳入；否则失败无法归因于 TorchPort 反馈粒度。

### 一个对照

5. **PraNet-V2。** 用作 positive/control，不计入 held-out 迁移成功率。它可以验证：相同权重和 352² 输入下，TorchPort 是否能验收官方 Jittor 端口、是否能发现故意注入的负例，以及 Logical Query Cost/Physical Evaluation Cost 记录是否正常。

## 共同阻塞项

- **权重供应链。** 除 BiRefNet 外，其余候选的官方 checkpoint 都只有 Google Drive/Baidu 页面。正式任务开始前必须由实验维护者离线取得、校验并放入 Agent 不可见但 evaluator 可读的 artifact 区；本调研没有执行下载。
- **数据可比性。** 只有 RUN 原生匹配四套 `Imgs/GT`；BiRefNet 和 VSCode-v2 需要只改路径的适配，PlantCamo 与 PraNet-V2 则是不同数据任务。跨任务只比较迁移成功率/成本时必须分层，不能把任务指标直接平均。
- **V100 预检。** 每个主候选都需要先做一次不计入正式实验的 source-only batch=1 smoke，记录峰值显存、依赖编译时间、完整权重加载和输出结构。本文所有 V100 结论目前都是未实测估计。
- **trace 预算。** BiRefNet、RUN 和 VSCode-v2 不适合默认全模块、全张量持久化。首轮应冻结可解释的 stage/block scope、调用次数上限和压缩策略；证据裁剪不能改变 `verify` 的最终判定。
- **外部端口污染。** PraNet-V2 已有官方 Jittor 实现，必须标为 control。其余四项在发放任务前还应再做一次带 commit 日期的 Jittor 端口复查；当前仓外“未发现”不是穷尽证明。
