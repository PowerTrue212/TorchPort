"""
Precision / amplification toy experiments for TorchPort.

Purpose
-------
Explain, with ONE consistent metric, why small floating-point differences
between two *equally correct* implementations can look huge in raw outputs,
and what controls that amplification (depth, normalization, weight scale,
input distribution, measurement stage).

Everything here is pure numpy. No torch, no GPU, no network calls, no git,
no src/ changes. Fixed seeds throughout for reproducibility.

Core definitions
-----------------
- Implementation A: float32 forward pass, matmul via np.dot (sequential/BLAS
  reduction order), i.e. whatever reduction order numpy's BLAS backend uses
  for the dot product along the contraction axis.
- Implementation B: float32 forward pass, SAME math (same weights, same
  biases, same activations, same normalization), but a different legal
  reduction order for every contraction: a manual pairwise (tree) summation
  computed in float32. This is a different but equally valid way to compute
  the same mathematical dot product.
- Reference R: float64 forward pass using implementation A's code path
  (sequential dot), used as ground truth for "what the real number is".

Metric everywhere unless noted: relative L2 error per layer,
    err(X, Y) = ||X - Y||_2 / ||Y||_2
computed on the per-layer activation tensors (flattened).

Run:
    E:\\Anaconda\\python.exe run_experiments.py
Outputs:
    figs/*.png, results.json, console timing.
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 20261007
EPS = 1e-30  # guard against div-by-zero in relative error denominators


# ---------------------------------------------------------------------------
# Utilities: reductions, metrics
# ---------------------------------------------------------------------------

def pairwise_sum(x: np.ndarray, axis: int) -> np.ndarray:
    """Pairwise (tree) summation along `axis`, computed in x's own dtype.

    This is a legitimate, different reduction order from a straight
    sequential/BLAS sum. For float32 inputs it accumulates in float32,
    so it exhibits different rounding than a sequential float32 sum,
    while remaining mathematically the same sum.
    """
    x = np.moveaxis(x, axis, 0)
    n = x.shape[0]
    if n == 1:
        return np.moveaxis(x, 0, axis) if False else x[0]
    # Recursive pairwise split, preserving dtype at every step.
    mid = n // 2
    left = pairwise_sum(x[:mid], axis=0) if mid > 1 else x[0]
    right = pairwise_sum(x[mid:], axis=0) if (n - mid) > 1 else x[mid]
    return (left + right).astype(x.dtype)


def sequential_dot(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """a: (N, in), w: (in, out) -> (N, out) using np.dot (BLAS reduction order)."""
    return np.dot(a, w)


def pairwise_dot(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Same contraction as sequential_dot but reduced via manual pairwise sum
    along the contraction axis, in the same dtype as the inputs.

    a: (N, in), w: (in, out) -> (N, out)
    """
    dtype = a.dtype
    # Broadcast multiply: (N, in, 1) * (1, in, out) -> (N, in, out), then
    # pairwise-sum over the "in" axis (axis=1).
    prod = (a[:, :, None].astype(dtype) * w[None, :, :].astype(dtype)).astype(dtype)
    return pairwise_sum(prod, axis=1)


def relerr(x: np.ndarray, ref: np.ndarray) -> float:
    """Relative L2 error of x against ref, flattened, as float64 scalars."""
    x64 = x.astype(np.float64).ravel()
    r64 = ref.astype(np.float64).ravel()
    num = np.linalg.norm(x64 - r64)
    den = np.linalg.norm(r64)
    return float(num / max(den, EPS))


def mae(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean(np.abs(x.astype(np.float64) - ref.astype(np.float64))))


# ---------------------------------------------------------------------------
# A small MLP with per-layer activation capture, run under three "engines":
#   engine="A64": float64, sequential dot      -> reference R
#   engine="A":   float32, sequential dot      -> implementation A
#   engine="B":   float32, pairwise dot        -> implementation B
# Optional LayerNorm-like normalization after each hidden linear+activation.
# ---------------------------------------------------------------------------

def gelu_erf(x):
    # exact erf-based GELU
    from math import sqrt
    return 0.5 * x * (1.0 + erf_np(x / np.sqrt(x.dtype.type(2.0))))


def erf_np(x):
    # numpy has no erf; use math.erf vectorized via scipy-free approximation
    # Abramowitz-Stegun 7.1.26, good to ~1.5e-7, applied elementwise in the
    # array's own dtype to keep float32 experiments genuinely float32.
    dtype = x.dtype
    sign = np.sign(x)
    ax = np.abs(x)
    a1, a2, a3, a4, a5 = (dtype.type(v) for v in
                          (0.254829592, -0.284496736, 1.421413741,
                           -1.453152027, 1.061405429))
    p = dtype.type(0.3275911)
    t = dtype.type(1.0) / (dtype.type(1.0) + p * ax)
    y = dtype.type(1.0) - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return (sign * y).astype(dtype)


def gelu_tanh(x):
    # tanh approximation of GELU (the deliberately WRONG variant used in Exp D)
    dtype = x.dtype
    c = dtype.type(0.7978845608028654)  # sqrt(2/pi)
    k = dtype.type(0.044715)
    inner = c * (x + k * x ** 3)
    return dtype.type(0.5) * x * (dtype.type(1.0) + np.tanh(inner))


def layernorm(x, gamma, beta, eps):
    dtype = x.dtype
    eps = dtype.type(eps)
    mu = x.mean(axis=-1, keepdims=True).astype(dtype)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True).astype(dtype)
    xn = (x - mu) / np.sqrt(var + eps)
    return (xn * gamma + beta).astype(dtype)


class ToyMLP:
    """A ~16-layer MLP: Linear -> GELU -> [optional LayerNorm], repeated,
    final Linear -> logits. Weights are shared across engines/dtypes (cast
    per call), so "A vs B" differ ONLY in reduction order, never in weights.
    """

    def __init__(self, dims, use_norm: bool, gelu_fn=gelu_erf, ln_eps=1e-5, seed=0):
        self.dims = dims
        self.use_norm = use_norm
        self.gelu_fn = gelu_fn
        self.ln_eps = ln_eps
        rng = np.random.default_rng(seed)
        self.weights = []  # list of (W, b, gamma, beta) in float64 master copies
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            self.weights.append({"fan_in": fan_in, "fan_out": fan_out})
        self._rng = rng

    def init_weights(self, scale_mode: str, seed: int):
        """scale_mode: 'he', 'xavier', 'too_large'. Stores float64 master
        weights; each engine casts them to its working dtype at call time."""
        rng = np.random.default_rng(seed)
        params = []
        n_layers = len(self.dims) - 1
        for i in range(n_layers):
            fan_in, fan_out = self.dims[i], self.dims[i + 1]
            if scale_mode == "he":
                std = np.sqrt(2.0 / fan_in)
            elif scale_mode == "xavier":
                std = np.sqrt(1.0 / fan_in)
            elif scale_mode == "too_large":
                std = np.sqrt(2.0 / fan_in) * 8.0
            else:
                raise ValueError(scale_mode)
            W = rng.standard_normal((fan_in, fan_out)).astype(np.float64) * std
            b = np.zeros((fan_out,), dtype=np.float64)
            gamma = np.ones((fan_out,), dtype=np.float64)
            beta = np.zeros((fan_out,), dtype=np.float64)
            params.append({"W": W, "b": b, "gamma": gamma, "beta": beta})
        self.params = params
        return params

    def set_trained_weights(self, params):
        self.params = params

    def forward(self, x, dtype, dot_fn, gelu_fn=None):
        """Run forward pass in given dtype using given dot_fn for every
        linear layer. Returns list of per-layer activations (post-activation,
        post-norm if used), including the final logits as the last entry.
        dtype: np.float32 or np.float64
        """
        gelu_fn = gelu_fn or self.gelu_fn
        acts = []
        h = x.astype(dtype)
        n_layers = len(self.params)
        for i, p in enumerate(self.params):
            W = p["W"].astype(dtype)
            b = p["b"].astype(dtype)
            z = dot_fn(h, W) + b
            is_last = (i == n_layers - 1)
            if not is_last:
                a = gelu_fn(z)
                if self.use_norm:
                    gamma = p["gamma"].astype(dtype)
                    beta = p["beta"].astype(dtype)
                    a = layernorm(a, gamma, beta, self.ln_eps)
                h = a
                acts.append(h)
            else:
                h = z  # raw logits, no activation
                acts.append(h)
        return acts


# ---------------------------------------------------------------------------
# Experiment A: summation order on a single float32 array
# ---------------------------------------------------------------------------

def blocked_sum(x: np.ndarray, block: int) -> np.float32:
    x = x.astype(np.float32)
    n = x.shape[0]
    partials = []
    for i in range(0, n, block):
        partials.append(np.sum(x[i:i + block], dtype=np.float32))
    partials = np.array(partials, dtype=np.float32)
    return np.sum(partials, dtype=np.float32)


def run_experiment_a(results: dict):
    print("[A] summation order experiment...")
    rng = np.random.default_rng(GLOBAL_SEED + 1)
    configs = [
        ("uniform_small_n1e4", lambda n: rng.standard_normal(n), 10_000),
        ("uniform_small_n1e6", lambda n: rng.standard_normal(n), 1_000_000),
        ("mixed_magnitude_n1e5", lambda n: rng.standard_normal(n) * (10.0 ** rng.integers(-3, 4, size=n)), 100_000),
        ("heavy_tail_n1e5", lambda n: rng.standard_cauchy(n), 100_000),
    ]
    out = {}
    for name, gen, n in configs:
        x64 = gen(n).astype(np.float64)
        x32 = x64.astype(np.float32)
        ref = np.sum(x64, dtype=np.float64)

        seq = np.sum(x32, dtype=np.float32)
        rev = np.sum(x32[::-1], dtype=np.float32)
        shuf_idx = rng.permutation(n)
        shuf = np.sum(x32[shuf_idx], dtype=np.float32)
        pw = pairwise_sum(x32, axis=0)
        blk64 = blocked_sum(x32, block=64)
        blk1024 = blocked_sum(x32, block=1024)

        def rel(v):
            return float(abs(float(v) - ref) / max(abs(ref), EPS))

        row = {
            "n": n,
            "ref_fp64": float(ref),
            "sequential": {"value": float(seq), "relerr": rel(seq)},
            "reversed": {"value": float(rev), "relerr": rel(rev)},
            "shuffled": {"value": float(shuf), "relerr": rel(shuf)},
            "pairwise": {"value": float(pw), "relerr": rel(pw)},
            "blocked_64": {"value": float(blk64), "relerr": rel(blk64)},
            "blocked_1024": {"value": float(blk1024), "relerr": rel(blk1024)},
        }
        spread = max(row[k]["relerr"] for k in
                     ("sequential", "reversed", "shuffled", "pairwise", "blocked_64", "blocked_1024"))
        row["max_relerr_vs_fp64"] = spread
        out[name] = row
        print(f"    {name}: n={n}, max relerr vs fp64 across orders = {spread:.3e}")

    # Figure: bar chart of relerr per order, per config (log scale)
    fig, ax = plt.subplots(figsize=(9, 5))
    orders = ["sequential", "reversed", "shuffled", "pairwise", "blocked_64", "blocked_1024"]
    width = 0.13
    xs = np.arange(len(configs))
    for i, order in enumerate(orders):
        vals = [max(out[name][order]["relerr"], 1e-20) for name, _, _ in configs]
        ax.bar(xs + (i - len(orders) / 2) * width, vals, width, label=order)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([name for name, _, _ in configs], rotation=20, ha="right")
    ax.set_ylabel("relative error vs float64 sum (log scale)")
    ax.set_title("Experiment A: float32 summation order vs float64 reference")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_a_summation_order.png", dpi=150)
    plt.close(fig)

    results["A"] = out


# ---------------------------------------------------------------------------
# Experiment B: 2x2 amplification study
# ---------------------------------------------------------------------------

def load_digits_data():
    try:
        from sklearn.datasets import load_digits
        d = load_digits()
        X = d.data.astype(np.float64) / 16.0  # scale to ~[0,1]
        y = d.target.astype(np.int64)
        return X, y, True
    except Exception:
        rng = np.random.default_rng(GLOBAL_SEED + 2)
        n, dim, ncls = 1200, 64, 10
        centers = rng.standard_normal((ncls, dim)) * 2.0
        y = rng.integers(0, ncls, size=n)
        X = centers[y] + rng.standard_normal((n, dim)) * 0.8
        X = (X - X.min()) / (X.max() - X.min())
        return X.astype(np.float64), y, False


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def one_hot(y, k):
    out = np.zeros((y.shape[0], k), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def train_mlp(mlp: ToyMLP, X, y, n_classes, seed, epochs=60, lr=0.05, batch_size=64):
    """Simple full-float64 SGD training (numpy autograd-free: manual
    backprop) to get a 'trained weights' condition quickly on CPU.
    Uses erf-GELU and plain linear algebra (sequential dot) throughout
    training; only forward-pass engines (A/B) differ at evaluation time.
    """
    rng = np.random.default_rng(seed)
    params = mlp.init_weights("he", seed=seed)
    n_layers = len(params)
    N = X.shape[0]

    def forward_train(x):
        cache = []
        h = x
        for i, p in enumerate(params):
            z = h @ p["W"] + p["b"]
            is_last = (i == n_layers - 1)
            if not is_last:
                pre_act = z
                a = gelu_erf(z)
                if mlp.use_norm:
                    mu = a.mean(axis=-1, keepdims=True)
                    var = ((a - mu) ** 2).mean(axis=-1, keepdims=True)
                    xn = (a - mu) / np.sqrt(var + mlp.ln_eps)
                    out = xn * p["gamma"] + p["beta"]
                    cache.append((h, z, a, mu, var, xn, out))
                    h = out
                else:
                    cache.append((h, z, a, None, None, None, a))
                    h = a
            else:
                cache.append((h, z, None, None, None, None, z))
                h = z
        return h, cache

    def backward_train(logits, y_batch, cache, lr):
        B = logits.shape[0]
        probs = softmax(logits)
        Y = one_hot(y_batch, logits.shape[1])
        dz = (probs - Y) / B  # dL/dlogits for softmax-CE

        for i in reversed(range(n_layers)):
            h_in, z, a, mu, var, xn, out = cache[i]
            p = params[i]
            is_last = (i == n_layers - 1)
            if is_last:
                dL_dz = dz
            else:
                dL_dout = dz
                if mlp.use_norm:
                    gamma = p["gamma"]
                    dgamma = np.sum(dL_dout * xn, axis=0)
                    dbeta = np.sum(dL_dout, axis=0)
                    dxn = dL_dout * gamma
                    D = xn.shape[-1]
                    std_inv = 1.0 / np.sqrt(var + mlp.ln_eps)
                    da = (1.0 / D) * std_inv * (
                        D * dxn - dxn.sum(axis=-1, keepdims=True)
                        - xn * (dxn * xn).sum(axis=-1, keepdims=True)
                    )
                    p["gamma"] -= lr * dgamma
                    p["beta"] -= lr * dbeta
                else:
                    da = dL_dout
                # GELU backward via finite-diff-free closed form using erf
                # d/dz [0.5 z (1+erf(z/sqrt2))] = 0.5(1+erf(z/sqrt2)) + z * phi(z)
                sqrt2 = np.sqrt(2.0)
                cdf = 0.5 * (1.0 + erf_np(z / sqrt2))
                pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
                dgelu = cdf + z * pdf
                dL_dz = da * dgelu

            dW = h_in.T @ dL_dz
            db = dL_dz.sum(axis=0)
            if i > 0:
                dz = dL_dz @ p["W"].T
            p["W"] -= lr * dW
            p["b"] -= lr * db

    idx_all = np.arange(N)
    for ep in range(epochs):
        rng.shuffle(idx_all)
        for s in range(0, N, batch_size):
            idx = idx_all[s:s + batch_size]
            xb, yb = X[idx], y[idx]
            logits, cache = forward_train(xb)
            backward_train(logits, yb, cache, lr)

    logits, _ = forward_train(X)
    preds = logits.argmax(axis=1)
    acc = float((preds == y).mean())
    return params, acc


def per_layer_errors(mlp: ToyMLP, x_input: np.ndarray, gelu_fn=None):
    """Run engines R(fp64,seq), A(fp32,seq), B(fp32,pairwise) and return
    per-layer relerr(A,R), relerr(B,R), relerr(A,B)."""
    actsR = mlp.forward(x_input, np.float64, sequential_dot, gelu_fn=gelu_fn)
    actsA = mlp.forward(x_input, np.float32, sequential_dot, gelu_fn=gelu_fn)
    actsB = mlp.forward(x_input, np.float32, pairwise_dot, gelu_fn=gelu_fn)
    n_layers = len(actsR)
    errAR, errBR, errAB = [], [], []
    for i in range(n_layers):
        errAR.append(relerr(actsA[i], actsR[i]))
        errBR.append(relerr(actsB[i], actsR[i]))
        errAB.append(relerr(actsA[i], actsB[i]))
    return errAR, errBR, errAB, actsA, actsB, actsR


def jacobian_gain_power_iter(mlp: ToyMLP, x_input_row: np.ndarray, n_iter=20, seed=0):
    """Estimate per-layer local gain = ||act_i(x+h v) - act_i(x)|| / ||h v||
    for a tiny random input-space perturbation v, in float64 for stability of
    the estimate itself (the gain is a property of the *mathematical*
    function, not of the engine's rounding).

    This is NOT a true Jacobian spectral norm (that would need backward /
    transpose-Jacobian power iteration). It is the simpler, honestly-named
    "max directional gain over n_iter random input directions", which only
    needs forward passes and is dimensionally safe. We report it as such in
    the README, not as a spectral norm.
    """
    rng = np.random.default_rng(seed)
    x = x_input_row.reshape(1, -1).astype(np.float64)
    acts0 = mlp.forward(x, np.float64, sequential_dot)
    n_layers = len(acts0)
    dims_in = x.shape[1]
    h = 1e-4
    gains_per_dir = []  # list of per-layer gain lists, one per random direction
    for _ in range(n_iter):
        v = rng.standard_normal((1, dims_in))
        v /= max(np.linalg.norm(v), EPS)
        acts1 = mlp.forward(x + h * v, np.float64, sequential_dot)
        in_pert_norm = np.linalg.norm(h * v)
        layer_gains = [float(np.linalg.norm(acts1[i] - acts0[i]) / max(in_pert_norm, EPS))
                       for i in range(n_layers)]
        gains_per_dir.append(layer_gains)
    gains_per_dir = np.array(gains_per_dir)  # (n_iter, n_layers)
    gains = gains_per_dir.max(axis=0).tolist()
    return gains


def run_experiment_b(results: dict):
    print("[B] 2x2 amplification study...")
    X, y, used_sklearn = load_digits_data()
    n_classes = int(y.max() + 1)
    in_dim = X.shape[1]
    rng = np.random.default_rng(GLOBAL_SEED + 3)

    # Split train/eval
    N = X.shape[0]
    idx = rng.permutation(N)
    n_train = int(N * 0.8)
    tr_idx, ev_idx = idx[:n_train], idx[n_train:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_ev, y_ev = X[ev_idx], y[ev_idx]

    hidden = 48
    n_hidden_layers = 14  # + 1 output layer => ~15 linear layers total
    dims = [in_dim] + [hidden] * n_hidden_layers + [n_classes]

    out = {"meta": {"used_sklearn_digits": used_sklearn, "dims": dims,
                     "n_train": int(len(tr_idx)), "n_eval": int(len(ev_idx))}}

    variants = {"no_norm": False, "with_norm": True}
    weight_modes = ["random_he", "random_xavier", "random_too_large", "trained"]
    input_modes = ["random_noise", "real_data"]

    cell_results = {}
    trained_cache = {}

    eval_n = min(64, len(ev_idx))
    x_real = X_ev[:eval_n]
    x_noise = rng.standard_normal((eval_n, in_dim))
    # scale noise to roughly match real-data norm so "random input" isn't
    # trivially out of distribution purely by scale
    x_noise = x_noise * (np.linalg.norm(x_real) / max(np.linalg.norm(x_noise), EPS))

    t0 = time.time()
    for norm_name, use_norm in variants.items():
        mlp = ToyMLP(dims, use_norm=use_norm, ln_eps=1e-5, seed=GLOBAL_SEED + 4)

        if ("trained", norm_name) not in trained_cache:
            print(f"    training MLP (use_norm={use_norm}) on digits ...")
            trained_params, acc = train_mlp(mlp, X_tr, y_tr, n_classes,
                                             seed=GLOBAL_SEED + 5, epochs=60,
                                             lr=0.08, batch_size=64)
            trained_cache[("trained", norm_name)] = (trained_params, acc)
            print(f"      train-set accuracy = {acc:.3f}")

        for weight_mode in weight_modes:
            if weight_mode == "trained":
                params, train_acc = trained_cache[("trained", norm_name)]
                mlp.set_trained_weights([{"W": p["W"].copy(), "b": p["b"].copy(),
                                           "gamma": p["gamma"].copy(), "beta": p["beta"].copy()}
                                          for p in params])
            else:
                scale_mode = {"random_he": "he", "random_xavier": "xavier",
                              "random_too_large": "too_large"}[weight_mode]
                mlp.init_weights(scale_mode, seed=GLOBAL_SEED + 6)
                train_acc = None

            for input_mode in input_modes:
                x_in = x_real if input_mode == "real_data" else x_noise
                errAR, errBR, errAB, actsA, actsB, actsR = per_layer_errors(mlp, x_in)
                gains = jacobian_gain_power_iter(mlp, x_in[0], n_iter=25, seed=GLOBAL_SEED + 7)

                key = f"{norm_name}|{weight_mode}|{input_mode}"
                cell_results[key] = {
                    "norm": norm_name, "weight_mode": weight_mode, "input_mode": input_mode,
                    "train_acc": train_acc,
                    "errAR_per_layer": errAR,
                    "errBR_per_layer": errBR,
                    "errAB_per_layer": errAB,
                    "gain_per_layer": gains,
                    "noise_envelope_max_per_layer": [max(a, b) for a, b in zip(errAR, errBR)],
                }
                print(f"    {key}: final-layer errAB={errAB[-1]:.3e}, "
                      f"noise envelope={max(errAR[-1], errBR[-1]):.3e}, gain[-1]={gains[-1]:.3e}")

    print(f"    experiment B wall time: {time.time() - t0:.1f}s")

    # Figures: err vs depth for 4 cells (random_he vs trained) x (noise vs real), per norm variant
    for norm_name in variants:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax_idx, weight_mode in enumerate(["random_he", "trained"]):
            ax = axes[ax_idx]
            for input_mode, style in zip(input_modes, ["-", "--"]):
                key = f"{norm_name}|{weight_mode}|{input_mode}"
                row = cell_results[key]
                depth = np.arange(1, len(row["errAB_per_layer"]) + 1)
                ax.plot(depth, np.clip(row["errAB_per_layer"], 1e-12, None), style,
                         label=f"err(A,B) [{input_mode}]", marker="o", markersize=3)
                ax.plot(depth, np.clip(row["noise_envelope_max_per_layer"], 1e-12, None), style,
                         label=f"max(err(A,R),err(B,R)) [{input_mode}]", marker="x", markersize=3, alpha=0.7)
            ax.set_yscale("log")
            ax.set_xlabel("layer depth")
            ax.set_title(f"{weight_mode}, norm={norm_name}")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("relative error (log scale)")
        axes[0].legend(fontsize=7, loc="upper left")
        fig.suptitle(f"Experiment B: err(A,B) tracks float32-vs-float64 noise envelope ({norm_name})")
        fig.tight_layout()
        fig.savefig(FIGS / f"fig_b_err_vs_depth_{norm_name}.png", dpi=150)
        plt.close(fig)

    # Gain vs depth figure (one figure, all weight_modes, norm on/off, real input)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax_idx, norm_name in enumerate(variants):
        ax = axes[ax_idx]
        for weight_mode in weight_modes:
            key = f"{norm_name}|{weight_mode}|real_data"
            row = cell_results[key]
            depth = np.arange(1, len(row["gain_per_layer"]) + 1)
            ax.plot(depth, np.clip(row["gain_per_layer"], 1e-6, None), marker="o", markersize=3, label=weight_mode)
        ax.set_yscale("log")
        ax.set_xlabel("layer depth")
        ax.set_title(f"norm={norm_name}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("local gain ||J*v|| / ||v|| (log scale)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Experiment B: estimated per-layer local gain vs depth")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_b_gain_vs_depth.png", dpi=150)
    plt.close(fig)

    out["cells"] = cell_results
    out["weight_modes"] = weight_modes
    out["input_modes"] = input_modes
    out["norm_variants"] = list(variants.keys())
    results["B"] = out
    return cell_results, dims, X_ev, y_ev, trained_cache


# ---------------------------------------------------------------------------
# Experiment C: output-measurement stage (logits -> probs -> 8-bit -> task metric)
# ---------------------------------------------------------------------------

def run_experiment_c(results: dict, dims, X_ev, y_ev, trained_cache):
    print("[C] output-measurement stage experiment...")
    rng = np.random.default_rng(GLOBAL_SEED + 8)
    n_classes = dims[-1]
    in_dim = dims[0]

    eval_n = min(80, len(X_ev))
    x_real = X_ev[:eval_n]
    x_noise = rng.standard_normal((eval_n, in_dim))
    x_noise = x_noise * (np.linalg.norm(x_real) / max(np.linalg.norm(x_noise), EPS))
    y_real = y_ev[:eval_n]

    out = {}
    for weight_mode in ["random_he", "trained"]:
        mlp = ToyMLP(dims, use_norm=True, ln_eps=1e-5, seed=GLOBAL_SEED + 4)
        if weight_mode == "trained":
            params, acc = trained_cache[("trained", "with_norm")]
            mlp.set_trained_weights([{"W": p["W"].copy(), "b": p["b"].copy(),
                                       "gamma": p["gamma"].copy(), "beta": p["beta"].copy()}
                                      for p in params])
        else:
            mlp.init_weights("he", seed=GLOBAL_SEED + 6)

        for input_mode, x_in, y_true in [("real_data", x_real, y_real), ("random_noise", x_noise, None)]:
            actsA = mlp.forward(x_in, np.float32, sequential_dot)
            actsB = mlp.forward(x_in, np.float32, pairwise_dot)
            logitsA, logitsB = actsA[-1], actsB[-1]

            probsA = softmax(logitsA.astype(np.float64))
            probsB = softmax(logitsB.astype(np.float64))

            # "8-bit map" stage: scale probability of predicted-max class to
            # [0,255] and round, mimicking x*255 rounded sigmoid maps.
            u8A = np.round(np.clip(probsA, 0, 1) * 255).astype(np.uint8)
            u8B = np.round(np.clip(probsB, 0, 1) * 255).astype(np.uint8)

            stage_logits = relerr(logitsA, logitsB)
            stage_probs = relerr(probsA, probsB)
            stage_u8 = relerr(u8A.astype(np.float64), u8B.astype(np.float64))
            stage_u8_mae = mae(u8A.astype(np.float64), u8B.astype(np.float64))

            predA = probsA.argmax(axis=1)
            predB = probsB.argmax(axis=1)
            pred_agreement = float((predA == predB).mean())
            task = {"pred_agreement_A_vs_B": pred_agreement}
            if y_true is not None:
                task["acc_A"] = float((predA == y_true).mean())
                task["acc_B"] = float((predB == y_true).mean())
                task["acc_diff"] = float(abs(task["acc_A"] - task["acc_B"]))

            key = f"{weight_mode}|{input_mode}"
            out[key] = {
                "stage_relerr": {
                    "raw_logits": stage_logits,
                    "probabilities": stage_probs,
                    "uint8_map": stage_u8,
                },
                "stage_mae": {
                    "uint8_map": stage_u8_mae,
                },
                "task_metrics": task,
            }
            print(f"    {key}: relerr logits={stage_logits:.3e} -> probs={stage_probs:.3e} "
                  f"-> uint8={stage_u8:.3e}; pred agreement={pred_agreement:.3f}")

    # Figure: stage relerr for each (weight_mode, input_mode), log scale
    fig, ax = plt.subplots(figsize=(9, 5))
    stages = ["raw_logits", "probabilities", "uint8_map"]
    keys = list(out.keys())
    width = 0.2
    xs = np.arange(len(stages))
    for i, key in enumerate(keys):
        vals = [max(out[key]["stage_relerr"][s], 1e-12) for s in stages]
        ax.bar(xs + (i - len(keys) / 2) * width, vals, width, label=key)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(stages)
    ax.set_ylabel("relative error between A and B (log scale)")
    ax.set_title("Experiment C: measurement stage shrinks/hides A-vs-B difference")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_c_measurement_stage.png", dpi=150)
    plt.close(fig)

    results["C"] = out


# ---------------------------------------------------------------------------
# Experiment D: bug vs noise
# ---------------------------------------------------------------------------

def run_experiment_d(results: dict, dims, X_ev, trained_cache):
    print("[D] bug vs noise experiment...")
    in_dim = dims[0]
    eval_n = min(64, len(X_ev))
    x_real = X_ev[:eval_n]

    bugs = {
        "ln_eps_1e-6": {"type": "ln_eps", "value": 1e-6},
        "gelu_tanh_approx": {"type": "gelu_tanh"},
        "wrong_norm_axis": {"type": "wrong_axis"},
    }

    out = {}
    for norm_name, use_norm in [("with_norm", True)]:
        for weight_mode in ["random_he", "trained"]:
            mlp = ToyMLP(dims, use_norm=use_norm, ln_eps=1e-5, seed=GLOBAL_SEED + 4)
            if weight_mode == "trained":
                params, acc = trained_cache[("trained", norm_name)]
                mlp.set_trained_weights([{"W": p["W"].copy(), "b": p["b"].copy(),
                                           "gamma": p["gamma"].copy(), "beta": p["beta"].copy()}
                                          for p in params])
            else:
                mlp.init_weights("he", seed=GLOBAL_SEED + 6)

            # Correct A, B, R
            errAR, errBR, errAB, actsA, actsB, actsR = per_layer_errors(mlp, x_real)
            envelope = [max(a, b) for a, b in zip(errAR, errBR)]

            for bug_name, bug_spec in bugs.items():
                if bug_spec["type"] == "ln_eps" and not use_norm:
                    continue  # eps bug meaningless without norm
                if bug_spec["type"] == "wrong_axis" and not use_norm:
                    continue

                mlp_bug = ToyMLP(dims, use_norm=use_norm,
                                  ln_eps=bug_spec.get("value", 1e-5), seed=GLOBAL_SEED + 4)
                mlp_bug.set_trained_weights([{"W": p["W"].copy(), "b": p["b"].copy(),
                                               "gamma": p["gamma"].copy(), "beta": p["beta"].copy()}
                                              for p in mlp.params])

                if bug_spec["type"] == "gelu_tanh":
                    actsBug = mlp_bug.forward(x_real, np.float32, pairwise_dot, gelu_fn=gelu_tanh)
                elif bug_spec["type"] == "wrong_axis":
                    # Monkey-patch: normalize over the WRONG axis (batch dim
                    # instead of feature dim) by transposing before/after.
                    def forward_wrong_axis(x_input, dtype, dot_fn):
                        acts = []
                        h = x_input.astype(dtype)
                        n_layers = len(mlp_bug.params)
                        for i, p in enumerate(mlp_bug.params):
                            W = p["W"].astype(dtype)
                            b = p["b"].astype(dtype)
                            z = dot_fn(h, W) + b
                            is_last = (i == n_layers - 1)
                            if not is_last:
                                a = gelu_erf(z)
                                gamma = p["gamma"].astype(dtype)
                                beta = p["beta"].astype(dtype)
                                # WRONG: normalize across batch axis (0) instead of feature axis (-1)
                                mu = a.mean(axis=0, keepdims=True).astype(dtype)
                                var = ((a - mu) ** 2).mean(axis=0, keepdims=True).astype(dtype)
                                xn = (a - mu) / np.sqrt(var + dtype(1e-5))
                                a = (xn * gamma + beta).astype(dtype)
                                h = a
                                acts.append(h)
                            else:
                                h = z
                                acts.append(h)
                        return acts
                    actsBug = forward_wrong_axis(x_real, np.float32, pairwise_dot)
                else:
                    actsBug = mlp_bug.forward(x_real, np.float32, pairwise_dot)

                errBugR = [relerr(actsBug[i], actsR[i]) for i in range(len(actsR))]
                margin = [errBugR[i] / max(envelope[i], EPS) for i in range(len(envelope))]

                # Fixed elementwise tolerance check, e.g. 1e-4 absolute on
                # final-layer activations (cast to float64 for comparison).
                tol = 1e-4
                final_bug_abs = np.abs(actsBug[-1].astype(np.float64) - actsR[-1].astype(np.float64))
                final_correctB_abs = np.abs(actsB[-1].astype(np.float64) - actsR[-1].astype(np.float64))
                flagged_bug = bool(np.max(final_bug_abs) > tol)
                flagged_correct = bool(np.max(final_correctB_abs) > tol)

                key = f"{weight_mode}|{bug_name}"
                out[key] = {
                    "errBugR_per_layer": errBugR,
                    "noise_envelope_per_layer": envelope,
                    "margin_bug_over_envelope_per_layer": margin,
                    "final_layer_margin": margin[-1],
                    "fixed_tol_1e-4": {
                        "flags_bug": flagged_bug,
                        "flags_correct_B": flagged_correct,
                        "max_abs_bug": float(np.max(final_bug_abs)),
                        "max_abs_correct_B": float(np.max(final_correctB_abs)),
                    },
                }
                print(f"    {weight_mode}|{bug_name}: final-layer margin over noise envelope = "
                      f"{margin[-1]:.2f}x; fixed-tol flags bug={flagged_bug}, flags correct B={flagged_correct}")

            out[f"{weight_mode}|__correct_B_baseline"] = {
                "errAB_per_layer": errAB,
                "noise_envelope_per_layer": envelope,
            }

    # Figure: per-layer error, bug vs noise envelope vs correct-B, for trained weights
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax_idx, weight_mode in enumerate(["random_he", "trained"]):
        ax = axes[ax_idx]
        baseline = out[f"{weight_mode}|__correct_B_baseline"]
        depth = np.arange(1, len(baseline["errAB_per_layer"]) + 1)
        ax.plot(depth, np.clip(baseline["noise_envelope_per_layer"], 1e-12, None), "k--",
                 marker="x", markersize=3, label="noise envelope max(err(A,R),err(B,R))")
        ax.plot(depth, np.clip(baseline["errAB_per_layer"], 1e-12, None), "g-",
                 marker="o", markersize=3, label="correct B: err(A,B)")
        for bug_name in bugs:
            key = f"{weight_mode}|{bug_name}"
            if key not in out:
                continue
            ax.plot(depth, np.clip(out[key]["errBugR_per_layer"], 1e-12, None),
                     marker="s", markersize=3, label=f"bug: {bug_name}")
        ax.set_yscale("log")
        ax.set_xlabel("layer depth")
        ax.set_title(weight_mode)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("relative error vs float64 reference (log scale)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Experiment D: injected bugs vs float32 noise envelope")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_d_bug_vs_noise.png", dpi=150)
    plt.close(fig)

    results["D"] = out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    np.random.seed(GLOBAL_SEED)
    results = {"seed": GLOBAL_SEED}

    run_experiment_a(results)
    cell_results, dims, X_ev, y_ev, trained_cache = run_experiment_b(results)
    run_experiment_c(results, dims, X_ev, y_ev, trained_cache)
    run_experiment_d(results, dims, X_ev, trained_cache)

    results_path = HERE / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print(f"Results written to: {results_path}")
    print(f"Figures written to: {FIGS}")


if __name__ == "__main__":
    main()
