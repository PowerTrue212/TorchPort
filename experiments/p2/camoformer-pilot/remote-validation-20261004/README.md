# CamoFormer COD remote validation

This directory contains compact evidence from the 2026-10-04 trusted GPU pilot. Raw tensor
archives, Jittor compilation caches, downloaded repositories, and the 1.4 GB trace run directory
remain on the worker and are intentionally excluded.

## Frozen setup

- GPU: NVIDIA GeForce RTX 3080 Ti, driver 580.105.08.
- PyTorch: 2.8.0+cu128; Jittor: 1.3.9.14.
- Source: `HVision-NKU/CamoFormer@5f924a91a13707b71ca1b986924504464ffd0638`.
- Independent calibration reference:
  `yinbow/CamoFormer-Jittor@be601810aa1e249440a5da281101bdfaae898cba`.
- Input: fixed float32 NCHW tensor with shape `[1, 3, 128, 128]`.
- Model contract: the complete PVTv2-B4 CamoFormer forward, returning P5 through P1.

The source CUDA run completed in 10.4627 seconds, produced all five outputs, and exported 1,100
portable state entries. The reference candidate loaded all 1,100 entries with no missing, extra, or
value-mismatched state and received the identical input.

## Result

The reference candidate is structurally valid but numerically fails:

- FP32: 79,772 / 131,072 compared elements mismatch (60.8612%), across all five output tensors.
- Strict: 81,757 / 131,072 elements mismatch (62.3756%).
- Maximum absolute error: 1560.2235.
- Normal target verification: one cached-oracle candidate execution, 40.8662 CUDA-configured
  process seconds.

Trace compared 1,195 like-named tensors. Strict first divergence is
`model.encoder.block1.0.attn.sr`, invocation 1, at trace position 5. That tensor has shape
`[1, 64, 4, 4]`, max absolute error 0.00133434, and 996 / 1,024 strict mismatches. The full trace
cost two CUDA-configured executions and 321.7928 process seconds; the target side accounted for
310.5287 seconds because it compiled trace-instrumented Jittor operators in a fresh trial cache.

The first reference attempt failed before model execution because `einops` was absent from the
target runtime. It was classified as `RUN_FAILED`, then the declared target dependency was installed
and the validation was rerun. This infrastructure attempt is not a model result.

This evidence establishes that the COD TaskSpec, state bridge, CUDA execution, numerical verdict,
trace, and localization all close. It does not establish migration success, and the known reference
must not be exposed to the Agent or counted as an unseen benchmark trial.

The original `runtime-snapshot.json` predates the manifest-level `runtime_packages` extension and
is kept unchanged as evidence. A second prepare-only pass produced `experiment-final.json`,
`run-plan-final.json`, and `runtime-snapshot-final.json`. The final snapshot records source
`timm 1.0.30`, `einops 0.8.2`, and `opencv-python-headless 4.14.0.94`; the target records
`einops 0.8.2`, while the other two packages are absent because the target adapter does not require
them. Its SHA256 is `1dedb67783c3f9ab20453beefe39d25c769a00ea148193949708c1ce0ef1e9e4`.

The supplemental snapshot was captured after the worker was restarted, on the same RTX 3080 Ti
with driver 595.71.05. It documents the environment used for subsequent Agent trials; the earlier
580.105.08 snapshot remains the environment record for the numerical calibration above.
