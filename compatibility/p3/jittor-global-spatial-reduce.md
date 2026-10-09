# Jittor 全局空间归约既不确定又比 PyTorch 粗约 10 倍；拆成两步归约后逐位确定

```json
{
  "schema_version": 1,
  "issue_id": "jittor-global-spatial-reduce",
  "title": "Jittor 对整张特征图的单次空间归约（AdaptiveAvgPool2d(1)、mean/sum over (H,W)）运行间不确定，且相对 fp64 误差约为 PyTorch 的 10 倍",
  "category": "numerical_backend",
  "status": "observed",
  "trigger": {
    "source_framework": "pytorch",
    "target_framework": "jittor",
    "source_version": "2.8.0+cu128",
    "target_version": "1.3.9.14",
    "device": "CUDA: RTX 3080 Ti, sm_86, CUDA 12.8",
    "code_pattern": "nn.AdaptiveAvgPool2d(1)(x), x.mean(dims=(2,3)), x.sum(dims=(2,3)) on a (1,64,176,176) float32 map; GroupNorm(8,64) in eval shows the same magnitude",
    "configuration": "conv_opt=1, use_cutlass=0"
  },
  "symptom": {
    "stage": "numerical",
    "failure_code": "PARITY_FAILED",
    "summary": "Tensors downstream of a global average pool exceed a k*D envelope although the implementation is correct; target run-to-run spread N is several times the source variant spread D."
  },
  "discrepancy": {
    "description": "Five repeats in one process: AdaptiveAvgPool2d(1) N_env 1.6e-6 and 1.4e-6 vs fp64 (PyTorch fp32 1.4e-7); sum over (H,W) N_env 2.1e-6; GroupNorm 1.6e-6. Convolutions are 1.5-1.8e-7; linear, matmul, softmax, LayerNorm, GELU, last-dim mean and bilinear interpolation are bitwise stable. In PlantCamo all 19 non-pass tensors of the eps-fixed copy were inside Att*.global_att, whose first layer is AdaptiveAvgPool2d(1). Probable cause is atomicAdd-merged partial sums (see jittor-reduce-nondeterminism); the generated kernel for this shape was not inspected."
  },
  "reproduction": {
    "fixture": "compatibility/fixtures/jittor-global-spatial-reduce/",
    "task": null,
    "instructions": [
      "python nenv_torch.py (PyTorch env) writes nenv_in.npz and fp64/fp32 references.",
      "python nenv_jt.py (Jittor env, conv_opt=1) runs each op 5 times; python nenv_cmp.py prints N_env and errors vs fp64.",
      "python nenv_reduce_alt.py compares four ways to write the global mean."
    ],
    "expected_failure": "avgpool_global and sum_spatial bitwise=False with N_env > 1e-6"
  },
  "repair": {
    "description": "Split the single 2D reduction or express it as a matmul: x.mean(dims=(3,)).mean(dims=(2,)) gave N_env 0 and 1.6e-7 vs fp64; matmul(x.reshape(B*C,H*W), ones/(H*W)) gave N_env 0 and 2.3e-7. Both match PyTorch fp32 accuracy on this shape.",
    "patch": null,
    "constraints": [
      "Verified on one shape only; the two-stage form is still a reduction and may become nondeterministic for other shapes.",
      "Not yet applied inside a full model and verified end to end."
    ]
  },
  "evidence": {
    "before": "experiments/p2/heldout-v1/results/plantcamo-full-r0/v3/pc-v3_evidence_noise_5583b9df1378b6e6.json",
    "after": null,
    "protocol_digest": null,
    "verified": false
  },
  "applicability": {
    "known": [
      "Jittor 1.3.9.14, RTX 3080 Ti sm_86, CUDA 12.8, conv_opt=1, shape (1,64,176,176).",
      "TorchPort noise.calibrate reproduces the same per-op values automatically (sum_spatial 2.17e-6 dominant)."
    ],
    "unknown": [
      "Dependence on spatial size, channel count, batch and GPU model.",
      "Whether conv_opt=0 or other Jittor flags change the reduction kernel.",
      "Whether the two-stage rewrite makes a full model pass v3 without a noise floor."
    ]
  },
  "provenance": {
    "source": "real_migration",
    "case_id": "plantcamo-full-r0"
  }
}
```
