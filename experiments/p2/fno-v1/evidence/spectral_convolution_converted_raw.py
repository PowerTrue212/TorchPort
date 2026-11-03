
import jittor as jt
from jittor import init
from jittor import nn
from typing import List, Optional, Tuple, Union
from ..utils import validate_scaling_factor
import tensorly as tl
from tensorly.plugins import use_opt_einsum
from .channel_mlp import ChannelMLP
from .einsum_utils import einsum_complexhalf
from .base_spectral_conv import BaseSpectralConv
from .resample import resample
tl.set_backend('pytorch')
use_opt_einsum('optimal')
einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def _contract_dense(x, weight, separable=False):
    order = tl.ndim(x)
    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])
    if separable:
        out_syms = ([x_syms[0]] + list(weight_syms))
    else:
        weight_syms.insert(1, einsum_symbols[order])
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    eq = f"{''.join(x_syms)},{''.join(weight_syms)}->{''.join(out_syms)}"
    if (not torch.is_tensor(weight)):
        weight = weight.to_tensor()
    if (x.dtype == torch.complex32):
        return einsum_complexhalf(eq, x, weight)
    else:
        return tl.einsum(eq, x, weight)

def _contract_dense_separable(x, weight, separable):
    if (not torch.is_tensor(weight)):
        weight = weight.to_tensor()
    return (x * weight)

def _contract_cp(x, cp_weight, separable=False):
    order = tl.ndim(x)
    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[(order + 1)]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [(einsum_symbols[1] + rank_sym)]
    else:
        out_syms[1] = out_sym
        factor_syms = [(einsum_symbols[1] + rank_sym), (out_sym + rank_sym)]
    factor_syms += [(xs + rank_sym) for xs in x_syms[2:]]
    eq = f"{x_syms},{rank_sym},{','.join(factor_syms)}->{''.join(out_syms)}"
    if (x.dtype == torch.complex32):
        return einsum_complexhalf(eq, x, cp_weight.weights, *cp_weight.factors)
    else:
        return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)

def _contract_tucker(x, tucker_weight, separable=False):
    order = tl.ndim(x)
    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[(order + 1):(2 * order)]
        factor_syms = [(xs + rs) for (xs, rs) in zip(x_syms[1:], core_syms)]
    else:
        core_syms = einsum_symbols[(order + 1):((2 * order) + 1)]
        out_syms[1] = out_sym
        factor_syms = [(einsum_symbols[1] + core_syms[0]), (out_sym + core_syms[1])]
        factor_syms += [(xs + rs) for (xs, rs) in zip(x_syms[2:], core_syms[2:])]
    eq = f"{x_syms},{core_syms},{','.join(factor_syms)}->{''.join(out_syms)}"
    if (x.dtype == torch.complex32):
        return einsum_complexhalf(eq, x, tucker_weight.core, *tucker_weight.factors)
    else:
        return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)

def _contract_tt(x, tt_weight, separable=False):
    order = tl.ndim(x)
    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])
    if (not separable):
        weight_syms.insert(1, einsum_symbols[order])
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[(order + 1):])
    tt_syms = []
    for (i, s) in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[(i + 1)]])
    eq = ((((''.join(x_syms) + ',') + ','.join((''.join(f) for f in tt_syms))) + '->') + ''.join(out_syms))
    if (x.dtype == torch.complex32):
        return einsum_complexhalf(eq, x, *tt_weight.factors)
    else:
        return tl.einsum(eq, x, *tt_weight.factors)

def get_contract_fun(weight, implementation='reconstructed', separable=False):
    "Generic ND implementation of Fourier Spectral Conv contraction\n\n    Parameters\n    ----------\n    weight : tensorly-torch's FactorizedTensor\n    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'\n        whether to reconstruct the weight and do a forward pass (reconstructed)\n        or contract directly the factors of the factorized weight with the input (factorized)\n    separable: bool\n        if True, performs contraction with individual tensor factors.\n        if False,\n    Returns\n    -------\n    function : (x, weight) -> x * weight in Fourier space\n    "
    if (implementation == 'reconstructed'):
        if separable:
            return _contract_dense_separable
        else:
            return _contract_dense
    elif (implementation == 'factorized'):
        if torch.is_tensor(weight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith('dense'):
                return _contract_dense
            elif weight.name.lower().endswith('tucker'):
                return _contract_tucker
            elif weight.name.lower().endswith('tt'):
                return _contract_tt
            elif weight.name.lower().endswith('cp'):
                return _contract_cp
            else:
                raise ValueError(f'Got unexpected factorized weight type {weight.name}')
        else:
            raise ValueError(f'Got unexpected weight type of class {weight.__class__.__name__}')
    else:
        raise ValueError(f'Got implementation={implementation}, expected "reconstructed" or "factorized"')
Number = Union[(int, float)]

class SpectralConv(BaseSpectralConv):
    'SpectralConv implements the Spectral Convolution component of a Fourier layer\n    described. \n    \n    It is implemented as described in [1]_ and [2]_.\n\n    Parameters\n    ----------\n    in_channels : int\n        Number of input channels\n    out_channels : int\n        Number of output channels\n    n_modes : int or int tuple\n        Number of modes to use for contraction in Fourier domain during training.\n\n        .. warning::\n\n            We take care of the redundancy in the Fourier modes, therefore, for an input\n            of size I_1, ..., I_N, please provide modes M_K that are I_1 < M_K <= I_N\n            We will automatically keep the right amount of modes: specifically, for the\n            last mode only, if you specify M_N modes we will use M_N // 2 + 1 modes\n            as the real FFT is redundant along that last dimension. See the theory guide for mode truncation details.\n\n\n        .. note::\n\n            Provided modes should be even integers. odd numbers will be rounded to the closest even number.\n\n        This can be updated dynamically during training.\n\n    complex_data : bool, optional\n        Whether data takes on complex values in the spatial domain, by default False.\n        If True, uses different logic for FFT contraction and uses full FFT instead of real-valued.\n    max_n_modes : int tuple or None, optional\n        If not None, maximum number of modes to keep in Fourier Layer along each dim\n        (n_modes cannot be increased beyond that). If None, all n_modes are used.\n        By default None.\n    bias : bool, optional\n        Whether to add a learnable bias to the output, by default True.\n    separable : bool, optional\n        Whether to use separable implementation of contraction.\n        If True, contracts factors of factorized tensor weight individually.\n        By default False.\n    resolution_scaling_factor : float, list of float, or None, optional\n        Scaling factor(s) for resolution scaling. If provided, the output resolution\n        will be scaled by this factor along each spatial dimension.\n        By default None.\n    fno_block_precision : str, optional\n        Precision mode for FNO block operations. Options: \'full\', \'half\', \'mixed\'.\n        By default \'full\'.\n    rank : float, optional\n        Rank of the tensor factorization of the Fourier weights, by default 1.0.\n        Ignored if ``factorization is None``.\n    factorization : str or None, optional\n        Tensor factorization type. Options: {\'tucker\', \'cp\', \'tt\'}.\n        If None, a single dense weight is learned for the FNO.\n        Otherwise, that weight, used for the contraction in the Fourier domain\n        is learned in factorized form. In that case, `factorization` is the\n        tensor factorization of the parameters weight used.\n        By default None.\n    implementation : {\'factorized\', \'reconstructed\'}, optional\n        If factorization is not None, forward mode to use:\n        * `reconstructed` : the full weight tensor is reconstructed from the\n          factorization and used for the forward pass\n        * `factorized` : the input is directly contracted with the factors of\n          the decomposition\n        Ignored if ``factorization is None``.\n        By default \'reconstructed\'.\n    enforce_hermitian_symmetry : bool, optional\n        Whether to enforce Hermitian symmetry conditions when performing inverse FFT\n        for real-valued data. When True, explicitly enforces that the 0th frequency\n        and Nyquist frequency are real-valued before calling irfft. \n        When False, relies on cuFFT\'s irfftn to handle symmetry automatically, \n        which may fail on certain GPUs or input sizes, causing line artifacts. \n        Setting to True splits the inverse FFT into ifftn along (n-1) dimensions \n        followed by irfft on the last dimension, with a small computational overhead. \n        By default True.\n    fixed_rank_modes : bool, optional\n        Modes to not factorize, by default False.\n        Ignored if ``factorization is None``.\n    decomposition_kwargs : dict or None, optional\n        Optional additional parameters to pass to the tensor decomposition.\n        Ignored if ``factorization is None``.\n        By default None.\n    init_std : float or \'auto\', optional\n        Standard deviation to use for weight initialization, by default \'auto\'.\n        If \'auto\', uses (2 / (in_channels + out_channels)) ** 0.5.\n    fft_norm : str, optional\n        FFT normalization parameter, by default \'forward\'.\n    device : torch.device or None, optional\n        Device to place the layer on, by default None.\n\n    References\n    ----------\n    .. [1] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential\n        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.\n        \n    .. [2] Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid\n        Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024).\n        TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.\n    '

    def __init__(self, in_channels, out_channels, n_modes, complex_data=False, max_n_modes=None, bias=True, separable=False, resolution_scaling_factor: Optional[Union[(Number, List[Number])]]=None, fno_block_precision='full', rank=1.0, factorization=None, implementation='reconstructed', enforce_hermitian_symmetry=True, fixed_rank_modes=False, decomposition_kwargs: Optional[dict]=None, init_std='auto', fft_norm='forward', device=None):
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.complex_data = complex_data
        self.n_modes = n_modes
        self.order = len(self.n_modes)
        if (max_n_modes is None):
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes
        self.fno_block_precision = fno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.implementation = implementation
        self.enforce_hermitian_symmetry = enforce_hermitian_symmetry
        self.resolution_scaling_factor: Union[(None, List[List[float]])] = validate_scaling_factor(resolution_scaling_factor, self.order)
        if (init_std == 'auto'):
            init_std = ((2 / (in_channels + out_channels)) ** 0.5)
        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None
        self.fft_norm = fft_norm
        if (factorization is None):
            factorization = 'Dense'
        if separable:
            if (in_channels != out_channels):
                raise ValueError(f'To use separable Fourier Conv, in_channels must be equal to out_channels, but got in_channels={in_channels} and out_channels={out_channels}')
            weight_shape = (in_channels, *max_n_modes)
        else:
            weight_shape = (in_channels, out_channels, *max_n_modes)
        self.separable = separable
        tensor_kwargs = (decomposition_kwargs if (decomposition_kwargs is not None) else {})
        self.weight = FactorizedTensor.new(weight_shape, rank=self.rank, factorization=factorization, fixed_rank_modes=fixed_rank_modes, **tensor_kwargs, dtype=torch.cfloat)
        init.gauss_(0, mean=init_std)
        self._contract = get_contract_fun(self.weight, implementation=implementation, separable=separable)
        if bias:
            self.bias = jt.array((init_std * torch.randn(*(tuple([self.out_channels]) + ((1,) * self.order)))))
        else:
            self.bias = None

    def transform(self, x, output_shape=None):
        in_shape = list(x.shape[2:])
        if ((self.resolution_scaling_factor is not None) and (output_shape is None)):
            out_shape = tuple([round((s * r)) for (s, r) in zip(in_shape, self.resolution_scaling_factor)])
        elif (output_shape is not None):
            out_shape = output_shape
        else:
            out_shape = in_shape
        if (in_shape == out_shape):
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        if (not self.complex_data):
            n_modes[(- 1)] = ((n_modes[(- 1)] // 2) + 1)
        self._n_modes = n_modes

    def _modulate_modes(self, x_hat, modes, condition_embedding=None):
        'Hook called on the kept spectral modes before the weight contraction.\n        The base implementation does nothing. Override in a subclass to apply\n        per-mode conditioning.\n        \n        See ConditionalSpectralConv)\n\n        Parameters\n        ----------\n        x_hat : torch.Tensor\n            Kept Fourier modes, shape (B, in_channels, *modes_shape).\n        modes: tuple of int\n            Shape of the kept-mode grid (same as x_hat.shape[2:]).\n        condition_embedding: torch.Tensor or None\n            Per-sample conditioning vector, shape (B, D).\n\n        Returns\n        -------\n        torch.Tensor\n            Modulated modes, same shape as x_hat.\n        '
        return x_hat

    def execute(self, x: torch.Tensor, output_shape: Optional[Tuple[int]]=None):
        return self._forward_impl(x, output_shape=output_shape)

    def _forward_impl(self, x: torch.Tensor, output_shape: Optional[Tuple[int]]=None, condition_embedding: Optional[torch.Tensor]=None):
        'Shared implementation used by forward and ConditionalSpectralConv.forward.'
        (batchsize, channels, *mode_sizes) = x.shape
        fft_size = list(mode_sizes)
        if (not self.complex_data):
            fft_size[(- 1)] = ((fft_size[(- 1)] // 2) + 1)
        fft_dims = list(range((- self.order), 0))
        if (self.fno_block_precision == 'half'):
            x = x.half()
        if self.complex_data:
            x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims[:(- 1)]
        if (self.order > 1):
            x = torch.fft.fftshift(x, dim=dims_to_fft_shift)
        if (self.fno_block_precision == 'mixed'):
            x = x.chalf()
        if (self.fno_block_precision in ['half', 'mixed']):
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat
        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size], device=x.device, dtype=out_dtype)
        starts = [(max_modes - min(size, n_mode)) for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)]
        if self.separable:
            slices_w = [slice(None)]
        else:
            slices_w = [slice(None), slice(None)]
        if self.complex_data:
            slices_w += [(slice((start // 2), ((- start) // 2)) if start else slice(start, None)) for start in starts]
        else:
            slices_w += [(slice((start // 2), ((- start) // 2)) if start else slice(start, None)) for start in starts[:(- 1)]]
            slices_w += [(slice(None, (- starts[(- 1)])) if starts[(- 1)] else slice(None))]
        slices_w = tuple(slices_w)
        weight = self.weight[slices_w]
        if self.separable:
            weight_start_idx = 1
        else:
            weight_start_idx = 2
        slices_x = [slice(None), slice(None)]
        for (all_modes, kept_modes) in zip(fft_size, list(weight.shape[weight_start_idx:])):
            center = (all_modes // 2)
            negative_freqs = (kept_modes // 2)
            positive_freqs = ((kept_modes // 2) + (kept_modes % 2))
            slices_x += [slice((center - negative_freqs), (center + positive_freqs))]
        if (weight.shape[(- 1)] < fft_size[(- 1)]):
            slices_x[(- 1)] = slice(None, weight.shape[(- 1)])
        else:
            slices_x[(- 1)] = slice(None)
        slices_x = tuple(slices_x)
        x_kept = self._modulate_modes(x[slices_x], tuple(weight.shape[weight_start_idx:]), condition_embedding)
        out_fft[slices_x] = self._contract(x_kept, weight, separable=self.separable)
        if ((self.resolution_scaling_factor is not None) and (output_shape is None)):
            mode_sizes = tuple([round((s * r)) for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)])
        if (output_shape is not None):
            mode_sizes = output_shape
        if (self.order > 1):
            out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:(- 1)])
        if self.complex_data:
            x = torch.fft.ifftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        elif self.enforce_hermitian_symmetry:
            out_fft = torch.fft.ifftn(out_fft, s=mode_sizes[:(- 1)], dim=fft_dims[:(- 1)], norm=self.fft_norm)
            out_fft[(..., 0)].imag.zero_()
            if ((mode_sizes[(- 1)] % 2) == 0):
                out_fft[(..., (- 1))].imag.zero_()
            x = torch.fft.irfft(out_fft, n=mode_sizes[(- 1)], dim=fft_dims[(- 1)], norm=self.fft_norm)
        else:
            x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        if (self.bias is not None):
            x = (x + self.bias)
        return x

class ConditionalSpectralConv(SpectralConv):
    "SpectralConv with per-mode conditioning via a learned (e, k) -> factor projection.\n\n    Inherits the full spectral convolution from SpectralConv. The only\n    addition is the _modulate_modes method, which multiplies the kept Fourier modes\n    by a per-mode complex factor derived from the conditioning embedding e\n    and a positional embedding of the mode indices phi_k.\n\n    Parameters\n    ----------\n    *args\n        Positional arguments forwarded to SpectralConv.\n\n    condition_embedding_channels : int\n        Width of the conditioning embedding e passed to forward.\n    modulation_type : {'real', 'complex', 'polar'}, optional\n        Output type of the per-mode factor. 'real' is a positive scalar,\n        'complex' is an unconstrained complex number, 'polar' is a\n        unit-magnitude phase rotation exp(i*theta). By default 'polar'.\n    k_embed_dim : int, optional\n        Dimension of the per-axis frequency-mode positional embedding. By default 32.\n    type_k : {'power', 'sinusoidal'}, optional\n        How to embed the mode index. By default 'power'.\n    hidden_channels : int, optional\n        Width of the hidden layer in the projection MLP. By default 64.\n    **kwargs\n        Keyword arguments forwarded to SpectralConv.\n    "

    def __init__(self, *args, condition_embedding_channels: int, modulation_type: str='polar', k_embed_dim: int=32, type_k: str='power', hidden_channels: int=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_embedding_channels = condition_embedding_channels
        self.modulation_type = modulation_type
        self.k_embed_dim = k_embed_dim
        self.type_k = type_k
        self._k_grid_cache = {}
        if (type_k == 'power'):
            self.register_buffer('k_powers', torch.linspace((- 2.0), 0.0, k_embed_dim))
        elif (type_k == 'sinusoidal'):
            num_freqs = ((k_embed_dim + 1) // 2)
            indices = torch.arange(0, num_freqs, dtype=torch.float32)
            self.register_buffer('k_inv_freqs', (10000.0 ** (((- 2.0) * indices) / k_embed_dim)))
        else:
            raise ValueError(f"unknown type_k: {type_k!r}. expected 'power' or 'sinusoidal'.")
        in_features = (condition_embedding_channels + (k_embed_dim * self.order))
        if (modulation_type in ('real', 'polar')):
            out_features = self.in_channels
        elif (modulation_type == 'complex'):
            out_features = (self.in_channels * 2)
        else:
            raise ValueError(f"unknown modulation_type: {modulation_type!r}. expected 'real', 'complex', or 'polar'.")
        self.modulator = ChannelMLP(in_channels=in_features, out_channels=out_features, hidden_channels=hidden_channels, n_dim=self.order)

    def _embed_k(self, shape: Tuple[(int, ...)], device=None) -> torch.Tensor:
        cache_key = (shape, device)
        if (cache_key not in self._k_grid_cache):
            n_dims = len(shape)
            k_ranges = []
            for (i, Si) in enumerate(shape):
                if (i < (n_dims - 1)):
                    half = (Si // 2)
                    k_ranges.append(torch.arange((- half), (Si - half), device=device))
                else:
                    k_ranges.append(torch.arange(0, Si, device=device))
            k_grid = torch.stack(torch.meshgrid(*k_ranges, indexing='ij'), dim=0).unsqueeze(0)
            self._k_grid_cache[cache_key] = k_grid
        k_grid = self._k_grid_cache[cache_key]
        n_dims = len(shape)
        if (self.type_k == 'power'):
            sign = torch.sign(k_grid)
            k_embed = (sign.unsqueeze(2) * (k_grid.abs().clamp_min(1.0).unsqueeze(2) ** self.k_powers.view((1, 1, (- 1), *([1] * n_dims)))))
        else:
            k_scaled = (k_grid.unsqueeze(2) * self.k_inv_freqs.view((1, 1, (- 1), *([1] * n_dims))))
            k_embed = jt.contrib.concat([torch.sin(k_scaled), torch.cos(k_scaled)], dim=2)
            k_embed = k_embed[(:, :, :self.k_embed_dim)]
        return 
        raise RuntimeError('origin source: <k_embed.reshape(1, (- 1), *shape)>,There are only 1 args in Pytorch reshape function, but you provide 3')

    def _modulate_modes(self, x_hat, modes, condition_embedding):
        if (condition_embedding is None):
            raise ValueError('ConditionalSpectralConv requires condition_embedding; got None.')
        if (condition_embedding.shape[1] != self.condition_embedding_channels):
            raise ValueError(f'condition_embedding width {condition_embedding.shape[1]} does not match condition_embedding_channels={self.condition_embedding_channels}.')
        B = condition_embedding.shape[0]
        kept_shape = tuple(x_hat.shape[2:])
        n_kept = len(kept_shape)
        t_feat = 
        raise RuntimeError('origin source: <condition_embedding.reshape(B, (- 1), *([1] * n_kept))>,There are only 1 args in Pytorch reshape function, but you provide 3').expand(B, (- 1), *kept_shape)
        k_feat = self._embed_k(kept_shape, device=x_hat.device).expand(B, (- 1), *kept_shape)
        mlp_out = self.modulator(jt.contrib.concat([t_feat, k_feat], dim=1))
        if (self.modulation_type == 'real'):
            factor = mlp_out
        elif (self.modulation_type == 'complex'):
            factor = torch.complex(mlp_out[(:, :self.in_channels)], mlp_out[(:, self.in_channels:)])
        else:
            factor = torch.exp((1j * mlp_out))
        return (x_hat * factor)

    def execute(self, x, output_shape=None, *, condition_embedding):
        return self._forward_impl(x, output_shape=output_shape, condition_embedding=condition_embedding)
