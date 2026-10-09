# PVT 类模型中 `partial(nn.LayerNorm, eps=1e-6)` 只对经 `norm_layer` 构造的层生效

```json
{
  "schema_version": 1,
  "issue_id": "pvt-layernorm-eps-partial",
  "title": "norm_layer=partial(nn.LayerNorm, eps=1e-6) 只作用于经 norm_layer 构造的层，直接写 nn.LayerNorm(...) 的层仍是 1e-5",
  "category": "api_semantics",
  "status": "observed",
  "trigger": {
    "source_framework": "pytorch",
    "target_framework": "jittor",
    "source_version": "2.8.0+cu128",
    "target_version": "1.3.9.14",
    "device": "CUDA: RTX 3080 Ti, sm_86",
    "code_pattern": "PVTv2 lib/pvtv2.py: pvt_v2_b* passes norm_layer=partial(nn.LayerNorm, eps=1e-6); Block.norm1/norm2 and PyramidVisionTransformerImpr.norm1..4 use norm_layer, while Attention.norm (sr_ratio>1) and OverlapPatchEmbed.norm call nn.LayerNorm(dim) directly",
    "configuration": "fp32 inference"
  },
  "symptom": {
    "stage": "numerical",
    "failure_code": "PARITY_FAILED",
    "summary": "Structure and weights pass (1278/1278); E1 first divergence at backbone.norm1 (rel 6.7e-6, about 5x the source envelope); downstream tensors fail. Task metrics barely move (S-measure diff 2e-7 on 250 test images), so E2 alone does not reveal it."
  },
  "discrepancy": {
    "description": "Two eps values coexist in the same backbone. In the PlantCamo full-r0 candidate the agent recognized both defaults but swapped the two groups: 12 Attention.norm layers were written with 1e-6 instead of 1e-5, and backbone.norm1..4 with 1e-5 instead of 1e-6 (17 modules). Module config comparison reports all 17 directly. The agent attributed the residual error to backend numerics instead."
  },
  "reproduction": {
    "fixture": "compatibility/fixtures/pvt-layernorm-eps-partial/cfg_probe.py",
    "task": "experiments/p2/heldout-v1/workspaces/_tasks/plantcamo-full-r0.json",
    "instructions": [
      "Dump source module config: python cfg_probe.py source src.json (PlantCamo source on sys.path).",
      "Dump candidate module config: python cfg_probe.py target <candidate repo> tgt.json (Jittor env).",
      "Compare: python cfg_probe.py cmp src.json tgt.json; expect eps mismatches on *.attn.norm and backbone.norm1..4."
    ],
    "expected_failure": "17 eps mismatches with used=true"
  },
  "repair": {
    "description": "Follow the construction path, not the class name: layers built through norm_layer get eps=1e-6, layers built with nn.LayerNorm(...) directly get PyTorch's default 1e-5. In the counterfactual copy, fixing the 5 module definitions (17 instances) reduced v2 max_ratio from 20.3 to 1.12 and failed tensors from 460 to 2; under v3 with noise_floor=2.17e-6 (offline rejudge) the copy passes with max ratio 1.55.",
    "patch": null,
    "constraints": [
      "The repair was applied by the evaluator in a counterfactual copy, not by the agent; it is not counted as the agent's result.",
      "Do not widen tolerance to absorb this error; it is a construction-parameter mismatch."
    ]
  },
  "evidence": {
    "before": "experiments/p2/heldout-v1/results/plantcamo-full-r0/v3/pc-v3_evidence_verify_agent.json",
    "after": "experiments/p2/heldout-v1/results/plantcamo-full-r0/v3/pc-v3_evidence_verify_cfeps.json",
    "protocol_digest": null,
    "verified": false
  },
  "applicability": {
    "known": [
      "PlantCamo MyNet with pvt_v2_b2 backbone (lib/pvtv2.py), Jittor 1.3.9.14.",
      "E2 on PlantCAMO1250 test (250 images): swapped-eps candidate S 0.8800298 vs PyTorch 0.8800299; changed pixel fraction 3.9e-5 vs 7.5e-6 for the fixed copy."
    ],
    "unknown": [
      "The after evidence passes only in an offline rejudge with noise_floor; an ordinary v3 verify run with noise_floor=auto has not been executed yet, so status stays observed.",
      "Other PVT-derived repositories (e.g. other COD models using pvtv2) are not yet checked; a second independent case is required for generalized."
    ]
  },
  "provenance": {
    "source": "real_migration",
    "case_id": "plantcamo-full-r0"
  }
}
```
