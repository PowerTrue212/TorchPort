# Metric Definitions

The main figure uses result-level metrics that an external reader can interpret
without knowing TorchPort internals.

| Metric | Meaning |
|---|---|
| RN | Runnable: the migrated Jittor project actually executed end to end. |
| NA | Numerical alignment: PyTorch and Jittor agreed on the final FP32 output within tolerance. |
| SA | Strict alignment: optional stricter intermediate-layer certificate also passed. |

Conversion-only counts are kept in `method_table.md` as supplementary evidence,
because syntactically valid translated code is not the same as a working
migration.
