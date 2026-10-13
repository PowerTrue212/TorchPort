# Jittor 归约波动机制图

图左读取 `docs/evidence/jittor_atomic_reduction/results.json`，展示同一输入重复 30 次的最大相对
L2 差异。图右只保留机制所需的三件事：多个线程、`atomicAdd` 和浮点加法不满足结合律。

- `Pool/Sum` 取空间求和和全局平均池化两者中的较大值。
- `Conv` 是 `conv_opt=1` 的 3×3 卷积。
- 原子加法和固定加法树计算同一个空间和；前者约为 `1.4e-6`，后者 30 次逐位一致。
图中不画横向连接线。所有普通文字使用同一字体，公式和科学计数法统一由 Matplotlib 数学排版
引擎渲染。

PNG 用于幻灯片，PDF 用于论文或矢量排版。运行：

```powershell
.\.venv\Scripts\python.exe experiments\figures\fig_jittor_atomic\plot.py
```
