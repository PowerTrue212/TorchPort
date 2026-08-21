# 脱敏实验结果

该目录只提交体积小、无凭据的结构化证据：

- `dsh_superres.json`：DSH 归一化最终报告。
- `dsh_superres_cert.json`：TorchPort 双端数值证书。
- `dsh_superres_trace.jsonl`：最终修复轮的工具轨迹。
- `official_superres.json`：同一 SuperRes 样本的官方 Jittor 转换器结果；代码转换与
  语法检查通过，但模型实例化失败。

模型副本、权重、NPZ 张量、虚拟环境和 DSH 会话原文不进入 Git。证书中的绝对路径
对应 `experiments/env.md` 记录的远端复现实验环境。
