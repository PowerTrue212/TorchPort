# Jittor avg_pool2d 拒绝窗口等于输入尺寸

```json
{
  "schema_version": 1,
  "issue_id": "jittor-avgpool-kernel-equals-input",
  "title": "Jittor avg_pool2d 在池化窗口等于输入空间尺寸时报错，PyTorch 正常返回",
  "category": "api_semantics",
  "status": "observed",
  "trigger": {
    "source_framework": "pytorch",
    "target_framework": "jittor",
    "source_version": "2.8.0+cu128",
    "target_version": "1.3.9.14",
    "device": "CUDA sm_86 (RTX 3080 Ti)",
    "code_pattern": "F.avg_pool2d(x, (H, 1)) where H equals the input height",
    "configuration": null
  },
  "symptom": {
    "stage": "runtime",
    "failure_code": "RUN_FAILED",
    "summary": "RuntimeError: size of var should be larger than kernel_size"
  },
  "discrepancy": {
    "description": "PyTorch returns shape (N, C, 1, W) for kernel (H, 1) on an (N, C, H, W) input. Jittor 1.3.9.14 raises 'size of var should be larger than kernel_size' for the same call. The equivalent x.mean over the pooled axis with keepdim=True produces the same result in both frameworks."
  },
  "reproduction": {
    "fixture": "compatibility/fixtures/jittor-avgpool-kernel-equals-input",
    "task": null,
    "instructions": [
      "Run claims_jt.py with the Jittor environment from experiments/p2/heldout-v1/prompts/tools-full.md.",
      "Run claims_torch.py with the PyTorch environment.",
      "Compare the 'avg_pool2d kernel=(H,1)' lines."
    ],
    "expected_failure": "Jittor line reports RuntimeError; PyTorch line reports shape (1, 2, 1, 3)."
  },
  "repair": {
    "description": "Replace the pooling call with mean over the pooled axis with keepdim=True. Used by the VSCode-v2 full-r0 candidate, which passed protocol v3 verify (max_ratio 1.074).",
    "patch": null,
    "constraints": ["Only equivalent when the window covers the whole axis and there is no padding."]
  },
  "evidence": {
    "before": null,
    "after": "experiments/p2/heldout-v1/results/vscode-v2-full-r0/e1-verify-full.json",
    "protocol_digest": null,
    "verified": false
  },
  "applicability": {
    "known": ["jittor==1.3.9.14", "kernel equal to input size on one spatial axis"],
    "unknown": ["whether kernel equal to input on both axes behaves the same", "newer Jittor versions"]
  },
  "provenance": {
    "source": "real_migration",
    "case_id": "vscode-v2-full-r0"
  }
}
```

## 同一 Agent 报告中其余说法的复核（不入库）

VSCode-v2 full-r0 的 Agent 报告了另外几条“Jittor 约束”。用同一 fixture 在相同环境下复核，均不成立或不是框架差异：

| Agent 的说法 | 复核结果 |
| --- | --- |
| `jt.matmul` 2-D×2-D 只返回最后一行，3-D 被折叠 | 不成立。2D×2D、3D×2D、3D×3D、4D×4D 形状正确，与 numpy 最大差 ≤ 4.8e-7 |
| `repeat_interleave` 后 `[..., 0, :]` 语义与 numpy 不同 | 不成立。形状正确，差 0 |
| `cat` 不支持 rank2 与 rank3 广播 | 两个框架都报错，不是 Jittor 特有 |
| `tuple(Var)` 会沿第 0 维切开 | 两个框架行为相同（`tuple(torch.Tensor)` 也沿第 0 维切开），是 adapter 接口约定，不是框架差异 |
