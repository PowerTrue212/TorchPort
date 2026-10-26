# OpenAI CLIP ViT-B/32：迁移到原生 Jittor

你在 OpenAI CLIP 官方仓库的独立 Git 工作区中工作。请把 ViT-B/32 模型迁移到原生 Jittor。
请先检查仓库、依赖、设备、数据和 checkpoint，再实现和测试。

在 `run.py` 实现 `load_model()`。它应返回原生 Jittor 模型。模型接收预处理后的图像和 token，
返回图像到文本、文本到图像的 logits。`evaluation_entry.py` 是固定入口，保持接口不变。
模型参数名及形状应与官方 ViT-B/32 checkpoint 一致。评测方加载预训练权重。

固定推理输入包括 CIFAR-10 测试图像及十个类别提示词。你可以使用正常的代码工具、测试、
网页检索和本会话提供的验证工具。安装依赖和存放缓存请使用 `.scratch/` 目录，它不属于交付内容。
完成后报告改动、验证结果、参考来源和仍未知项。
