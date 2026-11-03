"""Controller source entry for five deterministic Darcy FNO training steps."""
import os
import sys
import inspect
import textwrap
import types

if os.environ.get("TP_FNO_DEPS"):
    sys.path[:0] = os.environ["TP_FNO_DEPS"].split(os.pathsep)

import torch  # noqa: E402
from neuralop.models import FNO  # noqa: E402


class SourceFNO(FNO):
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        # Upstream adds a non-tensor metadata entry; TorchPort freezes tensors.
        state.pop("_metadata", None)
        return state

    def double(self) -> "SourceFNO":
        # Keep real/imaginary parts in FP64 throughout spectral calibration.
        self._apply(
            lambda value: value.to(torch.complex128) if value.is_complex()
            else value.to(torch.float64) if value.is_floating_point()
            else value
        )
        for conv in self.fno_blocks.convs:
            original = type(conv)._forward_impl
            source = textwrap.dedent(inspect.getsource(original))
            if source.count("out_dtype = torch.cfloat") != 1:
                raise RuntimeError("upstream spectral dtype branch changed")
            namespace = dict(original.__globals__)
            exec(source.replace("out_dtype = torch.cfloat", "out_dtype = x.dtype"), namespace)
            conv._forward_impl = types.MethodType(namespace[original.__name__], conv)
        return self


def load_train_model() -> SourceFNO:
    return SourceFNO(
        n_modes=(4, 4),
        in_channels=1,
        out_channels=1,
        hidden_channels=8,
        n_layers=2,
    )


def build_optimizer(network: FNO) -> torch.optim.Adam:
    return torch.optim.Adam(network.parameters(), lr=0.001)


def train_step(
    network: FNO,
    optimizer: torch.optim.Adam,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    loss = torch.mean((network(x) - y) ** 2)
    loss.backward()
    optimizer.step()
    return loss
