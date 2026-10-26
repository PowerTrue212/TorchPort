# SimCSE training seed preflight r0

The controller-only preflight completed on the V100. It did not start an
Agent trial. The source oracle passed both repeated runs and generated FP64
and alternate FP32 calibration. The untouched target seed failed when
`load_train_model()` raised its intended `NotImplementedError`. The formal
preflight `verify` result is `passed=false`, `failure_code=RUN_FAILED`.

The frozen TaskSpec is
`D:\科研\SimCSE-Benchmark-v1\controller\prep\task-train-seed-v1.json`.
`torchport inspect` accepted its two `[8,2,32]` training steps, 200-item
initial state archive, source entry, target entry, and `train-step` protocol.
The task digest is
`751203547d810b782fa01d835908d552a645b98700b5158d6f0e7a939c49b2bc`.

The raw result and remote execution report are in
`D:\科研\SimCSE-Benchmark-v1\controller\prep\train-seed-preflight-r0`.
The five source-oracle files were copied into its `source-oracle` directory.
Each local SHA-256 matches `oracle.json`. The local state file is a hard link
to the already saved initial-state archive with the same SHA-256. The total
logical size of the five files is 3,941,903,450 bytes. The copied oracle's
losses and eight selected post-step parameters match the standalone source
probe bitwise. See `copy-validation.json` and `probe-comparison.json` there.

The remote evidence directory uses about 5.0 GiB of data-disk space: about
3.7 GiB for the first source run and 1.3 GiB for its repeat. The oracle's
large files have link count two. This confirms that the oracle shares those
immutable files with the first source run. The data disk had about 2.6 GiB
free after the preflight. Its initial-state weight cache consumes additional
space outside that evidence directory. The V100 had sufficient GPU and host
memory for the FP32, FP64, and alternate FP32 source runs. Source construction
used 63.63 seconds of recorded execution time. The complete `verify` recorded
181.09 seconds across three processes, including the expected target failure.

The current `train-step` capture stores all 199 trainable tensors after each
step. One successful target run would add about 836 MiB. Two target repeats
would leave less than 1 GiB free. Normal Agent evaluation calls would then
exhaust this data disk. The full-parameter capture cannot support a trial with
the planned query budget on this V100. The next protocol must make its
observation set explicit. It must still load and check the complete initial
state and the full trainable-parameter set. The current preflight evidence
remains intact; no reduced checker result exists yet.

The local preflight script exited with a summary-printing `AttributeError`
after it saved `verify-result.json`. It expected the `oracle` field to be an
object; `verify` returned a path string. The script was corrected. This local
printing error did not affect the remote verification result or evidence.
