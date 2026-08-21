# Error Cards

这些错例记录 DSH 如何根据 TorchPort 证据选择通用修复，不是模型适配器。

## vit_pytorch round 0

- code: `CONVERT_FAILED`
- stage: `official_jittor`
- symptom: 官方转换器生成 `x[(:, 0)]`，入口文件语法检查失败
- repair: `none`

## resnet_cifar10 round 0

- code: `CONVERT_FAILED`
- stage: `official_jittor`
- symptom: 官方转换器生成的 resnet.py 第 35 行语法检查失败
- repair: `none`

## mae round 0

- code: `CONVERT_FAILED`
- stage: `official_jittor`
- symptom: 官方转换器不支持 nn.Parameter(..., requires_grad=False)，生成无效代码
- repair: `none`

## swin round 0

- code: `CONVERT_FAILED`
- stage: `official_jittor`
- symptom: 官方转换器不支持多参数 reshape 和 AdaptiveAvgPool1d，生成文件语法失败
- repair: `none`

## swin round 3

- code: `JITTOR_RUN_FAILED`
- stage: `align`
- symptom: jittor.nn 缺少 AdaptiveAvgPool1d
- repair: `compat.adaptive-pool`

## vit_pytorch round 1

- code: `TORCH_RUN_FAILED`
- stage: `align`
- symptom: from torch.nn import Module 导致 forward 未改为 execute
- repair: `module.forward-to-execute`

## vit_pytorch round 2

- code: `TORCH_RUN_FAILED`
- stage: `align`
- symptom: einops 不识别 jittor.Var
- repair: `compat.einops`

## vae round 1

- code: `JITTOR_CONVERT_FAILED`
- stage: `migration`
- symptom: 模型级验证被无关 train() 中的 loss.backward 阻塞
- repair: `scope.prune-code`

## vae round 2

- code: `JITTOR_RUN_FAILED`
- stage: `check`
- symptom: 模型切片裁掉 args 后仍保留 jt.set_global_seed(args.seed)
- repair: `scope.prune-code`

## vae round 4

- code: `TORCH_RUN_FAILED`
- stage: `align`
- symptom: 源端 argparse.parse_args 读取 TorchPort capture 参数
- repair: `probe.argv-isolation`

## vae round 5

- code: `SOURCE_SIDE_EFFECT`
- stage: `align`
- symptom: 源端加载原始训练脚本触发数据集和 DataLoader 顶层副作用
- repair: `source.model-scope`
