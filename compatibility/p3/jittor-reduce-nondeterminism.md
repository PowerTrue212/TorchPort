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
    "description": "On sm_70, five executions of the same input gave max_run_diff=7.153e-7 for a 1x1 Conv2d, 9.537e-7 for a 3x3 Conv2d, and 1.144e-5 for sum(dim=1). The generated sum kernel assigns different channel-strided partial sums to GPU threads and calls atomicAdd on the same output element; the generated conv kernel does the same for channel/kernel partial sums. Atomic addition makes each update indivisible but does not define the arrival order of updates from different threads. Because float32 addition is not associative, a different arrival order can change low bits. On sm_86, the CamoFormer evidence independently records repeat differences at both module and whole-network level. This entry does not claim that every reduction is nondeterministic: the same sm_70 probe's mean(dim=-1), depthwise Conv2d, Linear, and interpolation cases were bitwise stable for the tested shapes."
  },
  "reproduction": {
    "fixture": "CamoFormer/outputs/dsh-web-camoformer/probe/probe_jt_determinism.py",
    "task": "CamoFormer target repeatability",
    "instructions": [
      "Use Jittor 1.3.9.14 with CUDA enabled and set conv_opt=1 before importing Jittor.",
      "Run probe_jt_determinism.py in one process; it evaluates each operation five times with the same NumPy input and reports the largest difference from the first output.",
      "For the sm_70 confirmation, use the environment recorded in docs/development/remote-v100-env.md and preserve JITTOR_HOME and cache_path inside the probe directory.",
      "Inspect the generated *_op.cc files and confirm that the REDUCE=2 sum and the seven-dimensional Conv2d fused graph merge partial sums with atomicAdd."
    ],
    "expected_failure": "At least one tested float32 conv_opt=1 Conv2d or non-last-dimension sum has max_run_diff greater than zero."
  },
  "repair": {
    "description": "No end-to-end repair is verified. conv_opt=0 routes eligible convolution through cuDNN and made the isolated convolution probe bitwise stable on sm_86, but the CamoFormer whole-network result became worse for an unconfirmed reason. Disabling tensor cores, the tuner, or JIT kernel search did not remove the observed whole-network variation. A deterministic reduction implementation has not been tested.",
    "patch": null,
    "constraints": [
      "Do not treat a wider tolerance, fewer compared outputs, or averaging repeat results as a repair.",
      "Do not replace verify with trace or repeatability calibration; verify remains the only final acceptance decision.",
      "Do not assume conv_opt=0 is a safe workaround until the same frozen TaskSpec passes verify."
    ]
  },
  "evidence": {
    "before": "CamoFormer/outputs/dsh-web-camoformer/probe/trace-noise-calibrated-repeat-tolerance-fp32.json",
    "after": null,
    "protocol_digest": "sha256:4fc61e4eeb3f3ab07f453d75425af5cc9a7db6f3a88260cbeca3e5f9dbee25ab",
    "verified": false
  },
  "applicability": {
    "known": [
      "Jittor 1.3.9.14, NVIDIA GeForce RTX 3080 Ti, sm_86, CUDA 12.8: CamoFormer target repeatability varies at module and whole-network level.",
      "Jittor 1.3.9.14, Tesla V100-SXM2-16GB, sm_70, CUDA 11.3: direct probe reproduces nonzero differences for conv_opt=1 Conv2d and sum(dim=1), and the generated kernels contain atomicAdd.",
      "CamoFormer with conv_opt=1, whose ordinary Conv2d graph uses reindex, broadcast, multiply, and reduce(add)."
    ],
    "unknown": [
      "The exact generated sm_86 kernel was not preserved locally, so the atomicAdd code path was confirmed directly on sm_70 and inferred for the same Jittor 1.3.9.14 graph on sm_86.",
      "Which shapes, dtypes, reduction axes, thread schedules, drivers, or newer Jittor versions are bitwise stable.",
      "Why conv_opt=0 made isolated convolutions deterministic on sm_86 but worsened the full CamoFormer comparison.",
      "Whether a deterministic custom reduction or another backend setting can make the frozen end-to-end verify pass.",
      "The installed wheel declares AtomicTunerPass and SharedReducePass but omits their .cc implementations; the exact pass-internal rule that chooses atomic lowering remains unconfirmed."
    ]
  },
  "provenance": {
    "source": "real_migration",
    "case_id": "camoformer"
  }
}
```

## 源码和生成内核证据

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
  `CamoFormer/outputs/dsh-web-camoformer/probe/trace-noise-calibrated-repeat-tolerance-fp32.json`；
  汇总见 `CamoFormer/outputs/dsh-web-camoformer/README.md:206-231`。
- sm_70 / CUDA 11.3：
  `CamoFormer/outputs/dsh-web-camoformer/probe/trace-v100-baseline-r6.json` 与远端短探针输出；
  汇总见 `CamoFormer/outputs/dsh-web-camoformer/probe/v100-report.md:1-24,41-50`。

## 对重复 verify 的影响

固定 candidate、输入、权重和协议并不能保证 Jittor target 的输出逐位固定。若某些元素靠近
验收阈值，重复 `verify` 可能给出不同的 mismatch 数量，极端情况下也可能造成通过/失败抖动。
每次 `verify` 仍是该次实际执行的正式判定；不能挑选最好的一次、平均多次输出，或用 trace 的
噪声标定结果覆盖失败。实验报告应保留每次物理执行的原始证据，并把重复次数计入 Physical
Evaluation Cost。
