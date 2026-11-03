"""Probe Jittor's V100 pair-valued FFT forward, inverse, and gradient."""

import json
import sys
from pathlib import Path

import jittor as jt
import numpy as np
from jittor import nn

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
jt.flags.use_cuda = 1
rng = np.random.default_rng(20261030)
input_real = rng.standard_normal((1, 16, 16)).astype(np.float32)
reference = np.fft.fft2(input_real)

x = jt.array(input_real)
z = jt.stack([x, jt.zeros_like(x)], 3)
frequency = nn._fft2(z)
fft_real = frequency[:, :, :, 0].numpy()
fft_imag = frequency[:, :, :, 1].numpy()
np.savez_compressed(
    out / "fft-forward.npz",
    input=input_real,
    reference_real=reference.real.astype(np.float32),
    reference_imag=reference.imag.astype(np.float32),
    jittor_real=fft_real,
    jittor_imag=fft_imag,
)

recovered = nn._fft2(frequency, True)[:, :, :, 0]
loss = (recovered * recovered).mean()
gradient = jt.grad(loss, z)
recovered_np = recovered.numpy()
gradient_real = gradient[:, :, :, 0].numpy()
gradient_imag = gradient[:, :, :, 1].numpy()
expected_gradient = 2 * input_real / input_real.size
np.savez_compressed(
    out / "fft-backward.npz",
    recovered=recovered_np,
    gradient_real=gradient_real,
    gradient_imag=gradient_imag,
    expected_gradient_real=expected_gradient,
)
result = {
    "status": "pilot_only",
    "jittor": jt.__version__,
    "cuda_enabled": jt.flags.use_cuda,
    "public_jt_fft_available": hasattr(jt, "fft"),
    "nn_pair_fft_available": hasattr(nn, "_fft2"),
    "fft_max_abs_vs_numpy": float(
        np.max(np.abs((fft_real + 1j * fft_imag) - reference))
    ),
    "inverse_max_abs": float(np.max(np.abs(recovered_np - input_real))),
    "gradient_real_max_abs": float(
        np.max(np.abs(gradient_real - expected_gradient))
    ),
    "gradient_finite": bool(
        np.isfinite(gradient_real).all() and np.isfinite(gradient_imag).all()
    ),
}
(out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
