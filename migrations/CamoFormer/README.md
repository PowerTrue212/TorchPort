# CamoFormer migration evidence

This directory preserves the source checkout, independent calibration reference, dependencies, and
evidence from the first real COD migration. It is no longer a trial materializer.

The frozen upstream revisions are:

- PyTorch source: `HVision-NKU/CamoFormer@5f924a91a13707b71ca1b986924504464ffd0638`
- calibration reference: `yinbow/CamoFormer-Jittor@be601810aa1e249440a5da281101bdfaae898cba`

The reference is never copied into an Agent workspace. It was used only to calibrate the adapter,
state mapping, and tolerances. The resulting run is trusted pilot evidence, not an unseen benchmark.

The task is intentionally limited to the complete CamoFormer forward pass. Dataset loading, image
writing, training, and the repository's evaluation scripts are outside the migration contract.
TorchPort supplies one fixed NCHW float32 image, exports the deterministic PyTorch state, loads that
state into the candidate, and compares all five P5..P1 tensors.

The old direct-model-API materializer, minimal `submission.py` seed, and restricted-tool prompts were
removed. New trials use the full-repository workflow in
`experiments/p2/heldout-v1/workspaces/materialize.py`: the Agent receives an independent Git copy of
the source repository, keeps normal code tools, and delivers a frozen repository snapshot with a
thin `run.py` entry.

To reproduce the preserved CamoFormer evidence on a GPU worker, clone both revisions into
`source/HVision-NKU-CamoFormer` and `reference/CamoFormer-Jittor`, then install the source and target
dependencies into separate PyTorch/Jittor runtimes:

```bash
/path/to/torch-env/bin/python -m pip install -r migrations/CamoFormer/requirements-source.txt
/path/to/jittor-env/bin/python -m pip install -r migrations/CamoFormer/requirements-target.txt
```

The official upstream README also mentions Apex, but this inference-only evidence did not import or
require it. The frozen pilot reports and raw evidence entry points are under `outputs/`; do not
reinterpret them as results from the current Harness.
