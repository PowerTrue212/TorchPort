# RAFT v1 V100 experiment evidence

This directory is the compact, sanitized Git-facing evidence bundle. Large checkpoints, fixtures, references, workspaces, flow arrays, NPZ captures, and full session logs remain in V100/data-fs and are referenced by `manifest.json`.

- Codex: passed (RN=true, NA=true).
- pure DSH: stopped_failed (RN=true, NA=false); the final edited workspace was not evaluated.
- DSH + TorchPort: common FP32 evaluator passed; TorchPort strict remains false.

Missing agent time/token fields mean not recorded, never zero. See `audit.json` for the method-integrity and reporting caveats.
