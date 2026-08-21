# 实验环境

远端主机与项目：

```text
qx-slai3
/data/L202500448/torchport
```

源端环境：

```text
Python 3.10
PyTorch 2.6.0+cu124
/data/L202500448/qx/miniconda3/envs/qx/bin/python
```

目标端隔离环境：

```text
Jittor 1.3.11.0
NumPy 1.26.4
/data/L202500448/torchport/tools/jittor-env/bin/python
```

目标端单独固定 NumPy 1.26.4，避免 Jittor 1.3.11 与 NumPy 2 的数组桥接 ABI 不兼容。
TorchPort 在目标采集前执行已知数组往返自检，失败时拒绝签发数值证书。

DSH 环境：

```text
Node.js 22.23.2
@deepseek-ai/dsh CLI 0.1.0-rc.7
插件 API 0.1.0-rc.8
profile headless
```

状态：

```text
插件 bundle 加载       OK
真实工具调用            OK
SuperRes 端到端         verified
FP32                    true
Strict                  false
first                   conv3
Python tests            108 passed
DSH plugin tests        5 passed
```

API 凭据只通过进程环境注入，没有写入仓库、profile、轨迹或实验产物。
