# Jittor 官方转换器 baseline

目标是只评估官方静态转换器，不允许 DSH/Codex 修复。

协议：

```python
from jittor.utils.pytorch_converter import convert
converted = convert(pytorch_code)
```

每个入口文件转换后，继续检查：

```text
代码是否转换完成
目标端是否能 import
目标模型是否能实例化
Jittor forward 是否能运行
是否能加载同一权重
是否能通过数值对齐
```

任何失败都记录为结构化阶段，不人工修改：

```text
CONVERT_FAILED
IMPORT_FAILED
MODEL_INIT_FAILED
JITTOR_RUN_FAILED
WEIGHT_LOAD_FAILED
PARITY_FAILED
```
