
import jittor as jt
from jittor import init
from functools import partial, partialmethod
from typing import Tuple, List, Union, Literal
Number = Union[(float, int)]
from jittor import nn
import warnings
warnings.filterwarnings('once', category=UserWarning)
from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D
from ..layers.spectral_convolution import SpectralConv, ConditionalSpectralConv
from ..layers.padding import DomainPadding
from ..layers.fno_block import FNOBlocks, ConditionalFNOBlocks
from ..layers.channel_mlp import ChannelMLP
from ..layers.complex import ComplexValued
from .base_model import BaseModel

class FNO(BaseModel, name='FNO'):
    'N-Dimensional Fourier Neural Operator. The FNO learns a mapping between\n    spaces of functions discretized over regular grids using Fourier convolutions,\n    as described in [1]_.\n\n    The key component of an FNO is its SpectralConv layer (see\n    ``neuralop.layers.spectral_convolution``), which is similar to a standard CNN\n    conv layer but operates in the frequency domain.\n\n    For a deeper dive into the FNO architecture, refer to :ref:`fno_intro`.\n\n    Parameters\n    ----------\n    n_modes : Tuple[int, ...]\n        Number of modes to keep in Fourier Layer, along each dimension.\n        The dimensionality of the FNO is inferred from len(n_modes).\n        n_modes must be larger enough but smaller than max_resolution//2 (Nyquist frequency)\n    in_channels : int\n        Number of channels in input function. Determined by the problem.\n    out_channels : int\n        Number of channels in output function. Determined by the problem.\n    hidden_channels : int\n        Width of the FNO (i.e. number of channels).\n        This significantly affects the number of parameters of the FNO.\n        Good starting point can be 64, and then increased if more expressivity is needed.\n        Update lifting_channel_ratio and projection_channel_ratio accordingly since they are proportional to hidden_channels.\n    n_layers : int, optional\n        Number of Fourier Layers. Default: 4\n\n    Other Parameters\n    ----------------\n    lifting_channel_ratio : Number, optional\n        Ratio of lifting channels to hidden_channels.\n        The number of lifting channels in the lifting block of the FNO is\n        lifting_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels).\n    projection_channel_ratio : Number, optional\n        Ratio of projection channels to hidden_channels.\n        The number of projection channels in the projection block of the FNO is\n        projection_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels).\n    positional_embedding : Union[str, nn.Module], optional\n        Positional embedding to apply to last channels of raw input\n        before being passed through the FNO.\n        Options:\n        - "grid": Appends a grid positional embedding with default settings to the last channels of raw input.\n          Assumes the inputs are discretized over a grid with entry [0,0,...] at the origin and side lengths of 1.\n        - GridEmbeddingND: Uses this module directly (see :mod:`neuralop.embeddings.GridEmbeddingND` for details).\n        - GridEmbedding2D: Uses this module directly for 2D cases.\n        - None: Does nothing.\n        Default: "grid"\n    non_linearity : nn.Module, optional\n        Non-Linear activation function module to use. Default: F.gelu\n    norm : Literal["ada_in", "group_norm", "instance_norm"], optional\n        Normalization layer to use. Options: "ada_in", "group_norm", "instance_norm", None. Default: None\n    norm_groups : int, optional\n        Number of groups for GroupNorm, by default 1\n    complex_data : bool, optional\n        Whether the data is complex-valued. If True, initializes complex-valued modules. Default: False\n    use_channel_mlp : bool, optional\n        Whether to use an MLP layer after each FNO block. Default: True\n    channel_mlp_dropout : float, optional\n        Dropout parameter for ChannelMLP in FNO Block. Default: 0\n    channel_mlp_expansion : float, optional\n        Expansion parameter for ChannelMLP in FNO Block. Default: 0.5\n    channel_mlp_skip : Literal["linear", "identity", "soft-gating", None], optional\n        Type of skip connection to use in channel-mixing mlp. Options: "linear", "identity", "soft-gating", None.\n        Default: "soft-gating"\n    fno_skip : Literal["linear", "identity", "soft-gating", None], optional\n        Type of skip connection to use in FNO layers. Options: "linear", "identity", "soft-gating", None.\n        Default: "linear"\n    conv_bias_kernel : int, optional\n        Kernel size for the local convolutional bias term used when ``fno_skip="linear"``.\n        ``1`` preserves the default pointwise bias term, while larger kernels add a\n        local convolution alongside the global spectral convolution. Default: 1\n    resolution_scaling_factor : Union[Number, List[Number]], optional\n        Layer-wise factor by which to scale the domain resolution of function.\n        Options:\n        - None: No scaling\n        - Single number n: Scales resolution by n at each layer\n        - List of numbers [n_0, n_1,...]: Scales layer i\'s resolution by n_i\n        Default: None\n    domain_padding : Union[Number, List[Number]], optional\n        Percentage of padding to use.\n        Options:\n        - None: No padding\n        - Single number: Percentage of padding to use along all dimensions\n        - List of numbers [p1, p2, ..., pN]: Percentage of padding along each dimension\n        Default: None\n    fno_block_precision : str, optional\n        Precision mode in which to perform spectral convolution.\n        Options: "full", "half", "mixed". Default: "full". Default: "full"\n    stabilizer : str, optional\n        Whether to use a stabilizer in FNO block. Options: "tanh", None. Default: None.\n        stabilizer greatly improves performance in the case `fno_block_precision=\'mixed\'`.\n    max_n_modes : Tuple[int, ...], optional\n        Maximum number of modes to use in Fourier domain during training.\n        None means that all the n_modes are used.\n        Tuple of integers: Incrementally increase the number of modes during training.\n        This can be updated dynamically during training.\n    factorization : str, optional\n        Tensor factorization of the FNO layer weights to use.\n        Options: "None", "Tucker", "CP", "TT"\n        Other factorization methods supported by tltorch. Default: None\n    rank : float, optional\n        Tensor rank to use in factorization. Default: 1.0\n        Set to float <1.0 when using TFNO (i.e. when factorization is not None).\n        A TFNO with rank 0.1 has roughly 10% of the parameters of a dense FNO.\n    fixed_rank_modes : bool, optional\n        Whether to not factorize certain modes. Default: False\n    implementation : str, optional\n        Implementation method for factorized tensors.\n        Options: "factorized", "reconstructed". Default: "factorized"\n    decomposition_kwargs : dict, optional\n        Extra kwargs for tensor decomposition (see `tltorch.FactorizedTensor`). Default: {}\n    separable : bool, optional\n        Whether to use a separable spectral convolution. Default: False\n    preactivation : bool, optional\n        Whether to compute FNO forward pass with resnet-style preactivation. Default: False\n    conv_module : nn.Module, optional\n        Module to use for FNOBlock\'s convolutions. Default: SpectralConv\n    enforce_hermitian_symmetry : bool, optional\n        Whether to enforce Hermitian symmetry conditions when performing inverse FFT\n        for real-valued data. Only used when ``conv_module`` is :class:`SpectralConv`\n        or a subclass; ignored otherwise. When True, explicitly enforces that the 0th\n        frequency and Nyquist frequency are real-valued before calling irfft. When False,\n        relies on cuFFT\'s irfftn to handle symmetry automatically, which may fail on\n        certain GPUs or input sizes, causing line artifacts. By default True.\n\n    Examples\n    --------\n    >>> from neuralop.models import FNO\n    >>> model = FNO(n_modes=(12,12), in_channels=1, out_channels=1, hidden_channels=64)\n    >>> model\n    FNO(\n    (positional_embedding): GridEmbeddingND()\n    (fno_blocks): FNOBlocks(\n        (convs): SpectralConv(\n        (weight): ModuleList(\n            (0-3): 4 x DenseTensor(shape=torch.Size([64, 64, 12, 7]), rank=None)\n        )\n        )\n            ... torch.nn.Module printout truncated ...\n\n    References\n    ----------\n    .. [1] :\n\n    Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential\n        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.\n\n    '

    def __init__(self, n_modes: Tuple[(int, ...)], in_channels: int, out_channels: int, hidden_channels: int, n_layers: int=4, lifting_channel_ratio: Number=2, projection_channel_ratio: Number=2, positional_embedding: Union[(str, nn.Module)]='grid', non_linearity: nn.Module=F.gelu, norm: Literal[('ada_in', 'group_norm', 'instance_norm')]=None, norm_groups: int=1, complex_data: bool=False, use_channel_mlp: bool=True, channel_mlp_dropout: float=0, channel_mlp_expansion: float=0.5, channel_mlp_skip: Literal[('linear', 'identity', 'soft-gating', None)]='soft-gating', fno_skip: Literal[('linear', 'identity', 'soft-gating', None)]='linear', conv_bias_kernel: int=1, resolution_scaling_factor: Union[(Number, List[Number])]=None, domain_padding: Union[(Number, List[Number])]=None, fno_block_precision: str='full', stabilizer: str=None, max_n_modes: Tuple[(int, ...)]=None, factorization: str=None, rank: float=1.0, fixed_rank_modes: bool=False, implementation: str='factorized', decomposition_kwargs: dict=None, separable: bool=False, preactivation: bool=False, conv_module: nn.Module=SpectralConv, enforce_hermitian_symmetry: bool=True, block_factory=FNOBlocks):
        if (decomposition_kwargs is None):
            decomposition_kwargs = {}
        super().__init__()
        self.n_dim = len(n_modes)
        self._n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = int((lifting_channel_ratio * self.hidden_channels))
        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = int((projection_channel_ratio * self.hidden_channels))
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.conv_bias_kernel = conv_bias_kernel
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.complex_data = complex_data
        self.fno_block_precision = fno_block_precision
        if (positional_embedding == 'grid'):
            spatial_grid_boundaries = ([[0.0, 1.0]] * self.n_dim)
            self.positional_embedding = GridEmbeddingND(in_channels=self.in_channels, dim=self.n_dim, grid_boundaries=spatial_grid_boundaries)
        elif isinstance(positional_embedding, GridEmbedding2D):
            if (self.n_dim == 2):
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(f'Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}')
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif (positional_embedding is None):
            self.positional_embedding = None
        else:
            raise ValueError(f"Error: tried to instantiate FNO positional embedding with {positional_embedding},                              expected one of 'grid', GridEmbeddingND")
        if ((domain_padding is not None) and ((isinstance(domain_padding, list) and (sum(domain_padding) > 0)) or (isinstance(domain_padding, (float, int)) and (domain_padding > 0)))):
            self.domain_padding = DomainPadding(domain_padding=domain_padding, resolution_scaling_factor=resolution_scaling_factor)
        else:
            self.domain_padding = None
        if (resolution_scaling_factor is not None):
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = ([resolution_scaling_factor] * self.n_layers)
        self.resolution_scaling_factor = resolution_scaling_factor
        self.fno_blocks = block_factory(in_channels=hidden_channels, out_channels=hidden_channels, n_modes=self.n_modes, resolution_scaling_factor=resolution_scaling_factor, use_channel_mlp=use_channel_mlp, channel_mlp_dropout=channel_mlp_dropout, channel_mlp_expansion=channel_mlp_expansion, non_linearity=non_linearity, stabilizer=stabilizer, norm=norm, norm_groups=norm_groups, preactivation=preactivation, fno_skip=fno_skip, conv_bias_kernel=conv_bias_kernel, channel_mlp_skip=channel_mlp_skip, complex_data=complex_data, max_n_modes=max_n_modes, fno_block_precision=fno_block_precision, rank=rank, fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable, factorization=factorization, decomposition_kwargs=decomposition_kwargs, conv_module=conv_module, n_layers=n_layers, enforce_hermitian_symmetry=enforce_hermitian_symmetry)
        lifting_in_channels = self.in_channels
        if (self.positional_embedding is not None):
            lifting_in_channels += self.n_dim
        self.lifting = ChannelMLP(in_channels=lifting_in_channels, out_channels=self.hidden_channels, hidden_channels=self.lifting_channels, n_layers=2, n_dim=self.n_dim, non_linearity=non_linearity)
        if self.complex_data:
            self.lifting = ComplexValued(self.lifting)
        self.projection = ChannelMLP(in_channels=self.hidden_channels, out_channels=out_channels, hidden_channels=self.projection_channels, n_layers=2, n_dim=self.n_dim, non_linearity=non_linearity)
        if self.complex_data:
            self.projection = ComplexValued(self.projection)

    def execute(self, x, output_shape=None, **kwargs):
        "FNO's forward pass\n\n        1. Applies optional positional encoding\n\n        2. Sends inputs through a lifting layer to a high-dimensional latent space\n\n        3. Applies optional domain padding to high-dimensional intermediate function representation\n\n        4. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity)\n\n        5. If domain padding was applied, domain padding is removed\n\n        6. Projection of intermediate function representation to the output channels\n\n        Parameters\n        ----------\n        x : tensor\n            input tensor\n\n        output_shape : {tuple, tuple list, None}, default is None\n            Gives the option of specifying the exact output shape for odd shaped inputs.\n\n            * If None, don't specify an output shape\n\n            * If tuple, specifies the output-shape of the **last** FNO Block\n\n            * If tuple list, specifies the exact output-shape of each FNO Block\n        "
        if kwargs:
            warnings.warn(f'FNO.forward() received unexpected keyword arguments: {list(kwargs.keys())}. These arguments will be ignored.', UserWarning, stacklevel=2)
        return self._forward_impl(x, output_shape=output_shape)

    def _forward_impl(self, x, output_shape=None, condition_embedding=None):
        if (output_shape is None):
            output_shape = ([None] * self.n_layers)
        elif isinstance(output_shape, tuple):
            output_shape = (([None] * (self.n_layers - 1)) + [output_shape])
        if (self.positional_embedding is not None):
            x = self.positional_embedding(x)
        x = self.lifting(x)
        if (self.domain_padding is not None):
            x = self.domain_padding.pad(x)
        block_kwargs = ({} if (condition_embedding is None) else {'condition_embedding': condition_embedding})
        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx], **block_kwargs)
        if (self.domain_padding is not None):
            x = self.domain_padding.unpad(x)
        x = self.projection(x)
        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes

def partialclass(new_name, cls, *args, **kwargs):
    "Create a new class with different default values\n\n    See the Spherical FNO class in neuralop/models/sfno.py for an example.\n\n    Notes\n    -----\n    An obvious alternative would be to use functools.partial\n    >>> new_class = partial(cls, **kwargs)\n\n    The issue is twofold:\n    1. the class doesn't have a name, so one would have to set it explicitly:\n    >>> new_class.__name__ = new_name\n\n    2. the new class will be a functools object and one cannot inherit from it.\n\n    Instead, here, we define dynamically a new class, inheriting from the existing one.\n    "
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return type(new_name, (cls,), {'__init__': __init__, '__doc__': cls.__doc__, 'forward': cls.forward})

class ConditionalFNO(FNO, name='ConditionalFNO'):
    'FNO with parameter conditioning via spectral modulation and adaptive normalization FiLM.\n    Extends FNO by accepting a pre-computed condition_embedding in forward.\n\n    Parameters\n    ----------\n    condition_embedding_channels : int\n        Width of the conditioning embedding supplied to forward.\n    mode_modulation : bool, optional\n        Enable spectral per-mode modulation. Default: True.\n    film : bool, optional\n        Enable adaptive-normalization FiLM conditioning. Defaults True.\n    modulation_type, k_embed_dim, type_k, modulator_hidden_channels\n        Forwarded to ConditionalSpectralConv\n    All other parameters are inherited from FNO.\n    '

    def __init__(self, *args, condition_embedding_channels: int, mode_modulation: bool=True, film: bool=True, modulation_type: str='polar', k_embed_dim: int=32, type_k: str='power', modulator_hidden_channels: int=64, conv_module=None, **kwargs):
        if (conv_module is None):
            conv_module = (ConditionalSpectralConv if mode_modulation else SpectralConv)
        block_factory = partial(ConditionalFNOBlocks, condition_embedding_channels=condition_embedding_channels, mode_modulation=mode_modulation, film=film, modulation_type=modulation_type, k_embed_dim=k_embed_dim, type_k=type_k, modulator_hidden_channels=modulator_hidden_channels)
        super().__init__(*args, conv_module=conv_module, block_factory=block_factory, **kwargs)

    def execute(self, x, output_shape=None, *, condition_embedding, **kwargs):
        if kwargs:
            warnings.warn(f'ConditionalFNO.forward() received unexpected keyword arguments: {list(kwargs.keys())}.These arguments will be ignored.', UserWarning, stacklevel=2)
        return self._forward_impl(x, output_shape=output_shape, condition_embedding=condition_embedding)

class TFNO(FNO):
    'Tucker Tensorized Fourier Neural Operator (TFNO).\n\n    TFNO is an FNO with Tucker factorization enabled by default.\n\n    It uses Tucker factorization of the weights, making the forward pass efficient by contracting\n    directly with the factors of the decomposition.\n\n    This results in a fraction of the parameters of an equivalent dense FNO.\n\n    Parameters\n    ----------\n    factorization : str, optional\n        Tensor factorization method, by default "Tucker"\n    rank : float, optional\n        Tensor rank for factorization, by default 0.1.\n        A TFNO with rank 0.1 has roughly 10% of the parameters of a dense FNO.\n\n    All other parameters are inherited from FNO with identical defaults.\n    See FNO class docstring for the complete parameter list.\n\n    Examples\n    --------\n    >>> from neuralop.models import TFNO\n    >>> # Create a TFNO model with default Tucker factorization\n    >>> model = TFNO(n_modes=(12, 12), in_channels=1, out_channels=1, hidden_channels=64)\n    >>>\n    >>> # Equivalent FNO model with explicit factorization:\n    >>> model = FNO(n_modes=(12, 12), in_channels=1, out_channels=1, hidden_channels=64,\n    ...             factorization="Tucker", rank=0.1)\n    '

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('factorization', 'Tucker')
        kwargs.setdefault('rank', 0.1)
        super().__init__(*args, **kwargs)
