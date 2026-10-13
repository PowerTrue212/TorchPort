# Jittor CUDA 浮点归约的运行间不确定性

```json
{
  "schema_version": 1,
  "issue_id": "jittor-reduce-nondeterminism",
  "title": "Jittor CUDA 并行浮点归约可因 atomicAdd 顺序变化而产生运行间差异",
  "category": "numerical_backend",
  "status": "observed",
  "trigger": {
    "source_framework": "pytorch",
    "target_framework": "jittor",
    "source_version": null,
    "target_version": "1.3.9.14",
    "device": "CUDA: sm_86 / CUDA 12.8 and sm_70 / CUDA 11.3",
    "code_pattern": "float32 reduction whose generated CUDA kernel splits one output across threads and merges partial sums with atomicAdd; this includes conv_opt=1 Conv2d expressed as reindex + broadcast + multiply + reduce(add)",
    "configuration": "conv_opt=1; the direct sm_70 probe also covered use_tensorcore=0 and use_tensorcore=1"
  },
  "symptom": {
    "stage": "numerical",
    "failure_code": "TARGET_RUN_TO_RUN_MISMATCH",
    "summary": "The same inputs and parameters can produce different float32 output bits, mismatch counts, and maximum errors across target runs."
  },
  "discrepancy": {
    "description": "On sm_86, 30 executions of a fixed (1,64,176,176) input gave relative run-to-run differences of 1.84e-6 for sum over (H,W), 1.93e-6 for AdaptiveAvgPool2d(1), and 1.82e-7 for a conv_opt=1 3x3 Conv2d. Their preserved generated kernels all call atomicAdd. An explicit multi-block atomic sum varied by 1.39e-6, while the same sum implemented as a fixed shared-memory tree was bitwise identical across 30 runs. Three independent processes, 90 runs total, reproduced the split: all atomic paths varied and the fixed tree did not. On sm_70, five executions previously gave max_run_diff=7.153e-7 for a 1x1 Conv2d, 9.537e-7 for a 3x3 Conv2d, and 1.144e-5 for sum(dim=1). Atomic addition makes each update indivisible but does not define the arrival order of updates from different threads. Because float32 addition is not associative, a different arrival order can change low bits. This entry does not claim that every reduction is nondeterministic."
  },
  "reproduction": {
    "fixture": "docs/evidence/jittor_atomic_reduction/probe.py",
    "task": "CamoFormer target repeatability",
    "instructions": [
      "Use Jittor 1.3.9.14 with CUDA enabled and set conv_opt=1 and use_cutlass=0 before importing Jittor.",
      "Run probe.py with a dedicated JITTOR_HOME. It evaluates built-in sum, AdaptiveAvgPool2d(1), explicit atomic sum, fixed-tree sum, two-stage sum, and Conv2d 30 times on one fixed input.",
      "Repeat the process three times and run summarize_trials.py over the saved outputs.npz files.",
      "Inspect the preserved kernels under docs/evidence/jittor_atomic_reduction/kernels/."
    ],
    "expected_failure": "At least one tested float32 conv_opt=1 Conv2d or non-last-dimension sum has max_run_diff greater than zero."
  },
  "repair": {
    "description": "An isolated deterministic reduction is verified: one CUDA block per output and a fixed shared-memory tree produced identical bits in 90 runs across three processes, and reduced relative error versus fp64 from about 1e-6 to 2.18e-7. Splitting the spatial sum into two stages was also bitwise stable and reached 2.17e-7. No end-to-end model repair is verified. conv_opt=0 routes eligible convolution through cuDNN and made the isolated convolution probe bitwise stable on sm_86, but the CamoFormer whole-network result became worse for an unconfirmed reason.",
    "patch": null,
    "constraints": [
      "Do not treat a wider tolerance, fewer compared outputs, or averaging repeat results as a repair.",
      "Do not replace verify with trace or repeatability calibration; verify remains the only final acceptance decision.",
      "Do not assume conv_opt=0 is a safe workaround until the same frozen TaskSpec passes verify."
    ]
  },
  "evidence": {
    "before": "migrations/CamoFormer/outputs/dsh-web-camoformer/probe/trace-noise-calibrated-repeat-tolerance-fp32.json",
    "after": "docs/evidence/jittor_atomic_reduction/results.json",
    "protocol_digest": "sha256:b047cdcf7990ac884b0cdbaa9b27180390baf401c8cb4c9cacdb5df1e6d01b8e",
    "verified": false
  },
  "applicability": {
    "known": [
      "Jittor 1.3.9.14, NVIDIA GeForce RTX 3080 Ti, sm_86, CUDA 12.8: exact generated kernels for global sum, AdaptiveAvgPool2d(1), and conv_opt=1 Conv2d contain atomicAdd; the isolated causal control is preserved locally.",
      "Jittor 1.3.9.14, Tesla V100-SXM2-16GB, sm_70, CUDA 11.3: direct probe reproduces nonzero differences for conv_opt=1 Conv2d and sum(dim=1), and the generated kernels contain atomicAdd.",
      "CamoFormer with conv_opt=1, whose ordinary Conv2d graph uses reindex, broadcast, multiply, and reduce(add)."
    ],
    "unknown": [
      "Which shapes, dtypes, reduction axes, thread schedules, drivers, or newer Jittor versions are bitwise stable.",
      "Why conv_opt=0 made isolated convolutions deterministic on sm_86 but worsened the full CamoFormer comparison.",
      "Whether replacing the affected reductions can make the frozen end-to-end verify pass.",
      "The installed wheel declares AtomicTunerPass and SharedReducePass but omits their .cc implementations; the exact pass-internal rule that chooses atomic lowering remains unconfirmed."
    ]
  },
  "provenance": {
    "source": "real_migration",
    "case_id": "camoformer"
  }
}
```

## sm_86 因果对照（2026-10-11）

同一个 `(1,64,176,176)` float32 输入在 RTX 3080 Ti 上重复 30 次。随后再启动两个独立进程，
共得到 90 次输出。内置空间求和、全局平均池化和显式多块原子归约均发生逐位变化；固定归约树和
两步归约的 90 次输出全部逐位相同。

| 实现 | 30 次最大运行间相对差异 | 相对 fp64 最大误差 | 逐位一致 |
| --- | ---: | ---: | :---: |
| Jittor 空间求和 | 1.84e-6 | 1.29e-6 | 否 |
| Jittor 全局平均池化 | 1.93e-6 | 1.45e-6 | 否 |
| 显式 `atomicAdd` | 1.39e-6 | 9.98e-7 | 否 |
| 固定归约树 | 0 | 2.18e-7 | 是 |
| 两步求和 | 0 | 2.17e-7 | 是 |
| Jittor 3×3 卷积 | 1.82e-7 | 未计算 | 否 |

三次独立进程的第一次输出之间也呈现同样分界。完整结果见
`docs/evidence/jittor_atomic_reduction/results.json` 和 `cross_process.json`，原始归约输出保存在
`trials/*/outputs.npz`。

## 源码和生成内核证据

本次 sm_86 生成内核已保存在 `docs/evidence/jittor_atomic_reduction/kernels/`：

- `builtin_sum_spatial_op.cc:66-73` 和 `builtin_avgpool_op.cc:66-73`：各线程先计算局部和，再对同一
  输出地址调用 `atomicAdd`。
- `conv3x3_op.cc:140-147`：`conv_opt=1` 的融合卷积对通道和卷积核位置求局部和，再用
  `atomicAdd` 合并到输出。
- `explicit_atomic_op.cc:132-139`：因果对照中的显式多块原子归约。
- `fixed_tree_op.cc:103-120`：一个输出只交给一个线程块，按固定二叉树顺序合并共享内存中的
  部分和，没有 `atomicAdd`。

以下行号来自远端只读安装目录
`/root/data-fs/tp/target/lib/python3.12/site-packages/jittor`：

- `compile_extern.py:227-234`：正常 CUDA 初始化会加载 cuDNN；`conv_opt=1` 时库列表缩减为
  `cublas` 和 `curand`，因此不会注册 cuDNN 卷积。
- `nn.py:1001-1017`：非 depthwise、`groups == 1` 的 `Conv2d` 明确构造输入 `reindex`、权重
  `broadcast`、逐元素乘法和 `sum([2,5,6])`。
- `src/ops/reduce_op.cc:364-395`：`ReduceOp` 把归约轴写入 JIT key，并提供初始化与逐元素累加的
  原始循环语义。
- `src/opt/pass_manager.cc:113-116`：CUDA 代码生成会依次运行 `ParallelPass`、
  `AtomicTunerPass`、`SharedReducePass` 和 `FloatAtomicFixPass`。当前 wheel 只有
  `atomic_tuner_pass.h` 与 `shared_reduce_pass.h` 的声明，没有相应 `.cc`，所以不能仅从安装源码
  进一步确认是哪一个 pass 做出了具体 lowering 决策。

本次 sm_70 隔离探针把生成代码保存在
`/root/data-fs/tp/probe-g/home/.cache/jittor/.../cu11.3.109_sm_70/jit/`：

- `*hash_6448edbcc86db13_op.cc:12-18` 标明 `OP add`、`DIM 4`、`REDUCE 2`，即探针的
  `sum(dim=1)`；`44-75` 把同一归约轴分给多个线程，每个线程先算 `tmp0`，再对同一
  `op0_yp[op0_yid]` 调用 `atomicAdd`；`106-118` 启动该并行 kernel。
- `*hash_54fbdaf16e31d478_op.cc:26-55` 标明 fused graph 是 `broadcast_to + reindex + multiply +
  reduce(add)`；`87-147` 让不同线程分别处理输入通道，并以 `atomicAdd` 合并到卷积输出；
  `160-204` 把这段代码映射回四个 Jittor Op 并启动 kernel。
- `/root/data-fs/tp/probe-g/probe-output-isolated.txt:27-40` 保存短探针结果；脚本 SHA-256 为
  `4fc61e4eeb3f3ab07f453d75425af5cc9a7db6f3a88260cbeca3e5f9dbee25ab`，输出 SHA-256 为
  `4ece1df99f498ddb0cdd78d288adb096cb1623341a1bf7b1501c92e0108c9802`。

设备级证据分别为：

- sm_86 / CUDA 12.8：
  `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/trace-noise-calibrated-repeat-tolerance-fp32.json`；
  汇总见 `migrations/CamoFormer/outputs/dsh-web-camoformer/README.md:206-231`。
- sm_70 / CUDA 11.3：
  `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/trace-v100-baseline-r6.json` 与远端短探针输出；
  汇总见 `migrations/CamoFormer/outputs/dsh-web-camoformer/probe/v100-report.md:1-24,41-50`。

## 对重复 verify 的影响

固定 candidate、输入、权重和协议并不能保证 Jittor target 的输出逐位固定。若某些元素靠近
验收阈值，重复 `verify` 可能给出不同的 mismatch 数量，极端情况下也可能造成通过/失败抖动。
每次 `verify` 仍是该次实际执行的正式判定；不能挑选最好的一次、平均多次输出，或用 trace 的
噪声标定结果覆盖失败。实验报告应保留每次物理执行的原始证据，并把重复次数计入 Physical
Evaluation Cost。
