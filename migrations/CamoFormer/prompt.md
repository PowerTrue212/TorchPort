Migrate the target implementation so that it faithfully reproduces the source implementation.

The target seed intentionally contains only `torchport_adapter.py` and `submission.py`; create the
Jittor implementation files it needs. The public adapter instantiates `CamoFormer(None)`, so port
the complete PVT execution path and its five outputs. Training, datasets, image writing, test
scripts, and the unused Swin path are outside this task.

Start with `list_files` and `search_code`. Read only the relevant source files and use line ranges
for large files; do not look for duplicate PyTorch implementation files in the minimal target.
Modify only the target. Use TorchPort run and verification tools to assess the candidate. Work
autonomously until the candidate is ready or the experiment budget stops the trial. Do not ask for
shell access or hidden evaluator data.
