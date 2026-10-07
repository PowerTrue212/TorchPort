# 远端 V100 环境

更新时间：2026-10-07

## 结论

环境位于 `/root/data-fs/tp`。两个前缀均为带 `conda-meta` 的 Conda 环境，且均使用
Python 3.12.9：

| 用途 | Python 解释器 | 主要版本 |
| --- | --- | --- |
| PyTorch source | `/root/data-fs/tp/source/bin/python`（实际文件为 `python3.12`） | PyTorch 2.5.1+cu121、CUDA runtime 12.1、NumPy 2.2.6 |
| Jittor target | `/root/data-fs/tp/target/bin/python`（实际文件为 `python3.12`） | Jittor 1.3.9.14、NumPy 1.26.4 |

机器为 Tesla V100-SXM2-16GB，compute capability 7.0（sm_70），驱动
535.183.01。Jittor 使用系统 CUDA 11.3 的 nvcc 11.3.109；PyTorch wheel 自带
CUDA 12.1 runtime，驱动可以运行它。

## Jittor 运行环境

每次启动 Jittor 前设置以下变量：

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PATH=/usr/local/cuda-11.3/bin:/root/data-fs/tp/target/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export python_config_path=/root/data-fs/tp/target/bin/python3.12-config
export nvcc_path=/usr/local/cuda-11.3/bin/nvcc
export cache_path=/root/data-fs/tp/jittor-cache
export use_cutt=0
export use_cutlass=0
export use_mkl=0
```

`/root/data-fs/tp/target/bin/python3.12-config --prefix` 返回
`/root/data-fs/tp/target`。`use_cutt=0`、`use_cutlass=0` 和 `use_mkl=0` 关闭本次
CUDA Conv 不需要的可选扩展；远端下载出口会把这些扩展的下载内容替换成拦截页，导致
MD5 校验失败。Jittor 编译日志确认使用
`/usr/local/cuda-11.3/bin/nvcc`，CUDA key 为 `cu11.3.109_sm_70`，检测到的架构为
`[70]`。

## CUDA 验收

清理安装介质后重新运行了两端验收。输入为 0 到 8 的 `1x1x3x3` 张量，卷积核为全 1 的
`1x1x2x2` 张量。

- PyTorch：在 `cuda` 上调用 `torch.nn.functional.conv2d` 并执行
  `torch.cuda.synchronize()`。
- Jittor：设置 `jt.flags.use_cuda=1`，调用 `jt.nn.Conv2d` 并执行
  `jt.sync_all(True)`。

两端均输出：

```json
{"result": [[[[8.0, 12.0], [20.0, 24.0]]]], "sum": 64.0}
```

Jittor 另报告 `use_cuda=1`，实际 nvcc 路径为
`/usr/local/cuda-11.3/bin/nvcc`。两次命令远端退出码均为 0。

## 磁盘与安装说明

最终磁盘数据：

```text
/root/data-fs/tp                 5.6G
/root/.cache/jittor              24M
/root/data-fs 所在文件系统       879G total, 822G used, 12G available, 99%
```

远端 HTTPS 出口无法直接取得有效的 Conda/PyPI/GitHub 包，因此安装介质先在本机下载，
再只通过 `scripts/remote/ssh_exec.py put` 上传。安装后已删除远端 Miniconda 安装器、
wheelhouse、`/root/data-fs/tp/conda-pkgs`、source 的 Conda package cache，以及失败下载的
CUTT/MKL 文件；pip 安装全程使用 `--no-cache-dir`。未修改 apt、系统包或全局配置。
