# CamoFormer controlled migration pilot

This directory holds only the reproducible setup for the first COD harness run. Downloaded source,
the independent Jittor reference, generated fixtures, candidate workspaces, and trial outputs stay
local and are ignored by Git.

The frozen upstream revisions are:

- PyTorch source: `HVision-NKU/CamoFormer@5f924a91a13707b71ca1b986924504464ffd0638`
- calibration reference: `yinbow/CamoFormer-Jittor@be601810aa1e249440a5da281101bdfaae898cba`

The reference is never copied into an Agent workspace. It may be used only to calibrate the adapter,
state mapping, and tolerances. This makes the run suitable as a trusted harness smoke test, not as
an unseen benchmark result.

The task is intentionally limited to the complete CamoFormer forward pass. Dataset loading, image
writing, training, and the repository's evaluation scripts are outside the migration contract.
TorchPort supplies one fixed NCHW float32 image, exports the deterministic PyTorch state, loads that
state into the candidate, and compares all five P5..P1 tensors.

Each candidate seed is minimal: it contains only `torchport_adapter.py` and `submission.py`. The
PyTorch repository remains available through the read-only source tools and is not duplicated into
the writable target. The task prompt directs the Agent to the reachable PVT path and uses bounded
file reads so source exploration does not consume the budget twice.

On the GPU worker, clone both revisions into `source/HVision-NKU-CamoFormer` and
`reference/CamoFormer-Jittor`, install the source and target task dependencies into their already
separate PyTorch/Jittor runtimes, then materialize the experiment:

```bash
/path/to/torch-env/bin/python -m pip install -r CamoFormer/requirements-source.txt
/path/to/jittor-env/bin/python -m pip install -r CamoFormer/requirements-target.txt
```

The official README also mentions Apex, but this inference-only TaskSpec does not import or require
it. The materialized manifest adds `timm`, `einops`, and `opencv-python-headless` to
`runtime_packages`, so their exact installed versions are frozen in `runtime-snapshot.json`. Then
materialize the experiment:

```bash
python -m experiments.p2.prepare_camoformer \
  experiments/p2/camoformer-pilot \
  --source-python /path/to/torch-env/bin/python \
  --target-python /path/to/jittor-env/bin/python
```

Run `python -m experiments.harness.experiment ... --allow-trusted-pilot --prepare-only` before any
provider request, then run the same command without `--prepare-only`.
