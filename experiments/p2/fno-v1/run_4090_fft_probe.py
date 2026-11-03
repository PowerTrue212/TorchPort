"""Load cuFFT under conv_opt=1, then run the unchanged FNO FFT probe."""
import runpy
import sys

import jittor as jt

cache = jt.compiler.cache_path + "/cuda"
jt.compile_extern.setup_cuda_lib(
    "cufft", extra_flags=f' -L"{cache}" -llibcuda_extern '
)
print("cufft_ops_available", hasattr(jt.compile_extern, "cufft_ops"))
sys.argv = [sys.argv[1], sys.argv[2]]
runpy.run_path(sys.argv[0], run_name="__main__")
