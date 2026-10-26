# SimCSE training seed preflight r1

This controller-only preflight used the bounded observation TaskSpec. It did
not start an Agent trial. The source oracle passed two runs. It checked all
200 initial-state items and recorded all 199 trainable parameter names. It
compared 25 named parameters after both steps. The observed arrays contain
38,585,856 elements per step. With two losses, the oracle has 52 numerical
outputs. FP64 and alternate FP32 data exist for every output. The
calibration self-check passed with zero failed output tensors.

The untouched target seed again failed at `load_train_model()` with its
intended `NotImplementedError`. The preflight `verify` result is
`passed=false`, `failure_code=RUN_FAILED`. This result does not measure Jittor
training correctness. The controller report is in
`D:\科研\SimCSE-Benchmark-v1\controller\prep\train-seed-preflight-r1`.

The TaskSpec is
`D:\科研\SimCSE-Benchmark-v1\controller\prep\task-train-seed-observe-v1.json`.
Its digest is
`c75ce3534429e1e9f41f9948e95cbb171857c7d77e8e7a4c0714dad4f0e2444c`.
The observation list is [train-observe-v1.json](train-observe-v1.json). It
covers the word, position, and token-type embeddings; attention query weights
in all 12 layers; feed-forward outputs in layers 0, 5, and 11; selected
LayerNorm weights; and the SimCSE MLP. Other trainable names remain in the
exact set check. Their post-step values are not compared elementwise. The
specified `pass_k=8` is exploratory until a correct Jittor training reference
and negative controls validate it.

The five source-oracle files were copied to `source-oracle` in the controller
directory. Their SHA-256 values match `oracle.json`. The local initial-state
file shares storage with the already saved initial archive. The logical size
of these files is 1,672,849,250 bytes. The two losses and eight selected
post-step parameter arrays match the standalone source probe bitwise.
`copy-validation.json`, `probe-comparison.json`, and
`calibration-self-check.json` preserve those checks.

The remote preflight evidence uses about 2.3 GiB. The data disk had about
4.8 GiB free after the run. One target run would add about 295 MiB. Ten
evaluation calls and two final target repeats would add about 3.5 GiB before
logs and other task data. The current disk can support one such pilot if the
Agent does not create other large files. Two condition runs need evidence
archival and disk cleanup between them.

The previous full-output preflight r0 remains failed as expected. Its 29
remote evidence files were transferred to a local tar with per-file SHA-256
checks. The archive is
`D:\科研\SimCSE-Benchmark-v1\controller\prep\train-seed-preflight-r0\remote-evidence-85fd015ce88f.tar`.
It has 5,256,212,480 bytes and SHA-256
`37188768edeb40eb0bc296075e3f707c4fa0b40b1d6e531a1cb4158746329894`.
The corresponding remote files were removed after the archive was verified.
The original report and the copied full source oracle remain local.

The next step is a controller-only Jittor training reference. Use it to test
the frozen 25-parameter comparison and negative controls on this V100. Do not
start `none` or `full` Agent trials until the target tolerance and trial
stopping rule are frozen.
