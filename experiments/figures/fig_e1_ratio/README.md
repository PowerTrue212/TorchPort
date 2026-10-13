# E1 最大比值图

本目录把用户提供的 E1 最大比值图重画为论文风格，并同时输出高分辨率 PNG 和矢量 PDF。

```powershell
E:\Anaconda\python.exe experiments\figures\fig_e1_ratio\plot.py
```

## 数据边界

图中数值按用户提供的原图保留：四个正常候选为 `0.65`、`1.1`、`1.97`、`3.0`；四个已知错误为 `9`、`62`、`200`、`100000`。这张图用于复现该图的表达，不替代 `experiments/p2/heldout-v1/REPORT.md` 中当前协议版本的正式记录。

关于 Jittor 运行波动的 `atomicAdd` 机制解释、真实生成内核截图和固定顺序归约因果实验，需要等待远端 GPU 与源码证据补齐后另做，不包含在本图中。
