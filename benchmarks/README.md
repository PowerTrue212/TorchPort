# 基准测试

这里维护公开、固定版本且可重复下载的迁移测试集。

`entry_patterns/` 维护入口恢复覆盖率清单。它不是模型适配器集合，而是把已覆盖和待测
的入口模式固定成可统计对象，后续真实仓库压力测试直接按这些模式归因。

模型分为：

- 基础视觉网络：CNN、ResNet。
- 视觉 Transformer：ViT。
- 生成模型组件：UNet、DiT、VAE。
- 语言模型组件：注意力、旋转位置编码、KV Cache。
- 历史框架：TensorFlow SavedModel、Caffe prototxt 与 caffemodel。

每个模型条目必须记录仓库、提交或版本、下载文件、许可证、输入规格、参考输出、验证级别和预期资源。

下载文件统一放在 `artifacts/models`，不提交 Git。

## 已验证切片

- `cnn-resnet18`：PyTorch CPU → 原生 Jittor CPU，102 个权重直接映射。
- 输入、`conv1`、四个残差阶段、分类头和最终输出均保存为 NPZ 证据。
- `fp32-inference` 配置通过；`operator-strict` 配置首次分歧位于
  `layer1.0.conv1`。
- CUDA 与昇腾结果必须单独执行，不能用 CPU 结果代替。

- `repos/tiny`：仓库级单输入自动迁移与严格数值对齐。
- `repos/multi`：自动发现 `build()` 工厂、两个输入和 6 个权重；直接工厂路线与
  显式构造参数路线均完成严格数值对齐。
- `repos/struct`：自动发现位置字典中的 `left`、`right` 和关键字参数 `scale`；
  直接导入的两次 `relu` 与 6 个权重均完成严格数值对齐。
- `repos/nested`：自动发现元组输入，字典中的 `logits` 和嵌套元组中的两个中间
  输出均进入验收；11 个双端证据张量完成严格数值对齐。
- `repos/package`：入口、工厂和模型分别位于根目录与 `packnet`；导入别名、
  相对导入、4 个权重和 5 个证据张量完成严格数值对齐。
- `repos/weights`：在外部测试副本中生成带 `state_dict` 包装和 `module.` 前缀的
  检查点；源端与目标端均加载 4/4 权重。只含 1/4 参数的故障检查点保持数值一致，
  但被完整性门控正确拒绝。
- `repos/config`：从 JSON 和 YAML 读取并合并模型参数，通过 `**options` 创建模型；
  自动恢复 3 个构造参数，4 个权重和 5 个证据张量完成严格数值对齐。
- `repos/args`：从 `options()` 中的 `ArgumentParser` 恢复类型转换、`dest` 和布尔
  动作，得到 1 个位置参数与 3 个关键字参数；4 个权重和 5 个张量严格对齐。
- `repos/env`：从 `os.environ[]` 和 `os.getenv()` 恢复显式值与源码默认值；即使
  宿主机注入冲突值，4 个权重和 5 个张量仍完成严格数值对齐。
- `repos/dynamic`：模型可运行，但构造参数和输入形状来自动态对象；默认静态探针
  报告缺参，迁移阶段传入显式模型、构造参数和形状后，`auto` 完成迁移、运行
  检查和严格数值对齐。
- `pytorch/examples/super_resolution`：外部真实 PyTorch 示例仓库切片，只指定训练
  入口 `main.py`。系统从 README 示例命令恢复 required CLI 参数
  `upscale_factor=3`，从训练脚本静态发现 `Net(upscale_factor=...)`，并把可验证
  探针入口切到模型定义文件 `model.py`，避免执行训练循环和数据下载逻辑。原生
  Jittor 迁移自动处理 `torch.nn.init`、`Conv2d`、`PixelShuffle` 和 `forward`
  到 `execute`；目标端运行检查以 `probe-import` 通过，8/8 权重加载完整，10 个
  同名证据张量完成采集。`fp32-inference` 通过；`operator-strict` 首次数值分歧
  位于 `conv2`。
- `repos/module`：包模块和 `python -m modnet.train` 启动方式；README 中的
  多行命令通过反斜杠续行，系统恢复 required 参数 `width=5`、`hidden=7`，再把
  探针入口从 `modnet/train.py` 切到 `modnet/model.py`。
- `repos/configcmd`：README 示例命令提供 required `-c configs/net.yaml`；
  训练入口使用 `argparse`、`type=Path`、`open()` 和 `yaml.safe_load()` 读取配置，
  系统静态恢复模型参数 `hidden=6`、`classes=2`，并把探针入口切到 `model.py`。
- `repos/omega`：OmegaConf 风格配置入口，README 提供 `--config`，训练入口使用
  `OmegaConf.load(args.config)` 和 `cfg.model` 属性访问；系统静态读取 YAML 并恢复
  `width=4`、`hidden=6`，再从构造参数反推线性层输入形状 `1x4`。
- `repos/omegac`：OmegaConf 配置经 `OmegaConf.to_container(cfg.model, resolve=True)`
  转成普通字典后再 `**options` 构造模型；系统不引入 OmegaConf 运行依赖，仍能静态
  恢复 `width=4`、`hidden=6` 和 `1x4` 输入形状。
- `repos/archdict`：训练脚本使用 `models.__dict__[args.arch](...)` 从模型模块选择
  工厂；系统根据 README CLI 提示把 `args.arch` 还原为 `smallnet`，再把可验证入口
  切到 `models.py` 中的工厂。
- `repos/archget`：训练脚本先 `builder = getattr(models, args.arch)`，再调用
  `builder(...)`；系统跟踪赋值后的静态引用，恢复同一个 `smallnet` 工厂。
- `repos/timmfact`：`timm.create_model(args.model, ...)` 风格入口。系统恢复 README
  中的 required 参数，把工厂调用入口切到本地 `timm.py`，并根据工厂内部转发参数
  推断 `3x32x32` 输入。
- `repos/tvfact`：`torchvision.models.get_model(args.arch, ...)` 风格入口。系统把
  `from torchvision.models import get_model` 解析到模型包内的工厂，并保留实际工厂
  调用参数用于双端证书。
- `repos/hydrainst`：Hydra 风格 `instantiate(cfg.model)` 入口。系统从
  `OmegaConf.load(args.config)` 读取本地 YAML，解析 `_target_: model.Net`，直接把
  探针入口切到 `model.py`，不要求目标端安装或执行 Hydra。
- `repos/regbuild`：自定义注册表 `build_model(cfg.model)` 入口。构造函数内部使用
  `MODELS[type]` 和 `cfg.pop("type")`，系统静态解析注册表字典，把探针入口切到
  `registry.py`，同时用剩余配置推断模型输入形状。
- `repos/decreg`：`@MODELS.register_module()` 装饰器注册表。系统静态收集类装饰器
  生成的注册映射，解析 `MODELS[type]` 到本地模型类。
- `repos/deccall`：`@MODELS.register("tiny")` 命名装饰器注册表。系统使用装饰器
  参数作为注册键，覆盖 OpenMMLab/MMEngine 风格的轻量注册写法。
- `repos/cfgattr`：训练入口传入整份配置 `build_model(cfg)`，构造函数内部通过
  `cfg.model` 取模型配置；系统仍能解析注册表和剩余模型参数。
- `repos/cfgitem`：训练入口传入整份配置 `build_model(cfg)`，构造函数内部通过
  `cfg["model"]` 取模型配置；覆盖字典下标访问版本。
- `repos/yacsmerge`：YACS 风格 `CfgNode` 配置入口。系统静态执行 `CN()`、
  属性赋值、`merge_from_file()` 和 `merge_from_list()`，恢复合并后的注册表配置，
  再把探针入口切到 `registry.py`。
- `repos/yacsmod`：入口模块从 `config.py` 导入 `get_cfg()`。系统会用配置模块自己的
  静态环境执行本地函数，再继续处理入口模块里的 YACS 合并逻辑。
- `repos/dynimp`：配置中保存模型模块名和类名，注册表使用
  `importlib.import_module(...)` 与 `getattr(...)` 动态导入模型类。系统只在模块名和
  类名都能静态确定时解析成本地模型引用。
- `repos/launch`：`accelerate launch` 和 `torchrun -m` 启动命令。README 中的
  launcher 参数如 `--mixed_precision`、`--nproc_per_node` 被入口定位跳过；系统只
  恢复模型参数 `width=8`、`hidden=10`，并把探针入口切到 `launchpkg/model.py`。

- `vit-tiny-patch16`：PyTorch CPU → 原生 Jittor CPU，152 个权重直接映射。
- 保存 162 个张量，包括注意力分数、Softmax 概率、上下文、MLP 和块输出。
- `fp32-inference` 配置通过；`operator-strict` 首次分歧位于
  `blocks.0.attn.scores`，Softmax 后重新进入严格容差。
