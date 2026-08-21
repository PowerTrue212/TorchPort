from __future__ import annotations

from pathlib import Path

PORTLIB = '''from __future__ import annotations

import os
import math
import numpy as np
from types import SimpleNamespace

import jittor as jt
import jittor.transform as transform
from jittor import init as jt_init
from jittor import nn
from PIL import Image
from jittor.dataset import Dataset as JtData
from jittor.dataset import MNIST as JtMNIST
from jittor.optim import LambdaLR, Optimizer


class _Bridge(JtData):
    def __init__(self, source, **attrs):
        super().__init__()
        self.source = source
        self.set_attrs(total_len=len(source), **attrs)

    def __getitem__(self, index):
        return self.source[index]


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        **_ignored,
    ):
        self.dataset = dataset
        self.loader = _Bridge(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


class MNIST:
    def __init__(self, root, train=True, transform=None, download=False, **_ignored):
        data_root = str(root).rstrip(os.sep) + os.sep
        source = JtMNIST(data_root, train=train, download=download)
        self.data = source.mnist
        self.transform = transform

    def __getitem__(self, index):
        image = Image.fromarray(self.data["images"][index]).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, self.data["labels"][index]

    def __len__(self):
        return len(self.data["labels"])


class Adadelta(Optimizer):
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        super().__init__(params, lr)
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        for group in self.param_groups:
            group["squares"] = [jt.zeros_like(item).stop_grad() for item in group["params"]]
            group["deltas"] = [jt.zeros_like(item).stop_grad() for item in group["params"]]

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        for group in self.param_groups:
            lr = group.get("lr", self.lr)
            rho = group.get("rho", self.rho)
            eps = group.get("eps", self.eps)
            decay = group.get("weight_decay", self.weight_decay)
            states = zip(group["params"], group["grads"], group["squares"], group["deltas"])
            for param, grad, square, delta in states:
                if param.is_stop_grad():
                    continue
                grad = grad + decay * param
                square.update(rho * square + (1 - rho) * grad * grad)
                change = jt.sqrt(delta + eps) / jt.sqrt(square + eps) * grad
                delta.update(rho * delta + (1 - rho) * change * change)
                param.update(param - lr * change)
        self.post_step()


class StepLR(LambdaLR):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        rate = lambda epoch: gamma ** (epoch // step_size)
        super().__init__(optimizer, rate, last_epoch)


def calculate_gain(nonlinearity, param=None):
    if param is None:
        try:
            return jt_init.calculate_gain(nonlinearity)
        except TypeError:
            pass
    try:
        return jt_init.calculate_gain(nonlinearity, param)
    except TypeError:
        if nonlinearity == "relu":
            return math.sqrt(2.0)
        return 1.0


def constant_(tensor, value=0.0):
    return jt_init.constant_(tensor, value)


def kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    return jt_init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)


def kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    return jt_init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)


def xavier_uniform_(tensor, gain=1.0):
    return jt_init.xavier_uniform_(tensor, gain)


def normal_(tensor, mean=0.0, std=1.0):
    return jt_init.gauss_(tensor, mean, std)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return normal_(tensor, mean, std)


def global_avg_pool2d(value):
    return value.mean(dims=[2, 3], keepdims=True)


def argsort(value, dim=-1, descending=False):
    result = jt.argsort(value, dim=dim, descending=descending)
    if isinstance(result, tuple):
        return result[0]
    return result


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def to_2tuple(value):
    return _pair(value)


def identity(value):
    return value


def checkpoint(function, *args, **kwargs):
    return function(*args, **kwargs)


def _dims(value):
    return tuple(int(item) for item in value.shape)


def rearrange(value, pattern, **axes):
    pattern = " ".join(pattern.split())
    if pattern == "b c (h p1) (w p2) -> b (h w) (p1 p2 c)":
        batch, channels, height, width = _dims(value)
        p1 = int(axes["p1"])
        p2 = int(axes["p2"])
        h = height // p1
        w = width // p2
        value = value.reshape((batch, channels, h, p1, w, p2))
        value = value.transpose(0, 2, 4, 3, 5, 1)
        return value.reshape((batch, h * w, p1 * p2 * channels))
    if pattern == "b n (h d) -> b h n d":
        batch, tokens, channels = _dims(value)
        heads = int(axes["h"])
        return value.reshape((batch, tokens, heads, channels // heads)).transpose(0, 2, 1, 3)
    if pattern == "b h n d -> b n (h d)":
        batch, heads, tokens, dim = _dims(value)
        return value.transpose(0, 2, 1, 3).reshape((batch, tokens, heads * dim))
    raise NotImplementedError(f"unsupported rearrange pattern: {pattern}")


def repeat(value, pattern, **axes):
    pattern = " ".join(pattern.split())
    if pattern == "... d -> b ... d":
        batch = int(axes["b"])
        return value.unsqueeze(0).broadcast((batch,) + _dims(value))
    raise NotImplementedError(f"unsupported repeat pattern: {pattern}")


class Rearrange(nn.Module):
    def __init__(self, pattern, **axes):
        self.pattern = pattern
        self.axes = axes

    def execute(self, value):
        return rearrange(value, self.pattern, **self.axes)


class AdaptiveAvgPool1d(nn.Module):
    def __init__(self, output_size):
        self.output_size = _pick(output_size, 0)

    def execute(self, value):
        axis = len(value.shape) - 1
        length = value.shape[axis]
        if self.output_size == 1:
            return value.mean(dims=[axis], keepdims=True)
        parts = []
        for index in range(self.output_size):
            start = math.floor(index * length / self.output_size)
            end = math.ceil((index + 1) * length / self.output_size)
            parts.append(value[..., start:end].mean(dims=[axis], keepdims=True))
        return jt.concat(parts, dim=axis)


def _shape(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _pick(value, index):
    if isinstance(value, (tuple, list)):
        return value[index]
    return value


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        elementwise_affine=True,
        bias=True,
        **_ignored,
    ):
        self.normalized_shape = _shape(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = None
        self.bias = None
        if elementwise_affine:
            self.weight = jt.ones(self.normalized_shape, dtype="float32")
            if bias:
                self.bias = jt.zeros(self.normalized_shape, dtype="float32")

    def execute(self, value):
        start = len(value.shape) - len(self.normalized_shape)
        dims = list(range(start, len(value.shape)))
        mean = value.mean(dims=dims, keepdims=True)
        centered = value - mean
        var = (centered * centered).mean(dims=dims, keepdims=True)
        value = centered / jt.sqrt(var + self.eps)
        if self.weight is not None:
            value = value * self.weight
        if self.bias is not None:
            value = value + self.bias
        return value


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        **_ignored,
    ):
        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def execute(self, value):
        value = self.proj(value)
        if self.flatten:
            value = value.flatten(2).transpose(0, 2, 1)
        return self.norm(value)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        **_ignored,
    ):
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, value):
        batch, tokens, channels = value.shape
        qkv = self.qkv(value).reshape(
            (batch, tokens, 3, self.num_heads, channels // self.num_heads)
        )
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        query, key, val = qkv[0], qkv[1], qkv[2]
        attn = jt.matmul(query * self.scale, key.transpose(0, 1, 3, 2))
        attn = self.attn_drop(attn.softmax(dim=-1))
        value = jt.matmul(attn, val)
        value = value.transpose(0, 2, 1, 3).reshape((batch, tokens, channels))
        return self.proj_drop(self.proj(value))


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=None,
        drop=0.0,
        **_ignored,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        act_layer = act_layer or nn.GELU
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(_pick(drop, 0))
        self.fc2 = nn.Linear(hidden_features, out_features)

    def execute(self, value):
        value = self.drop(self.act(self.fc1(value)))
        return self.drop(self.fc2(value))


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        self.drop_prob = drop_prob

    def execute(self, value):
        return value


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=None,
        norm_layer=None,
        **_ignored,
    ):
        norm_layer = norm_layer or nn.LayerNorm
        act_layer = act_layer or nn.GELU
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def execute(self, value):
        value = value + self.drop_path(self.attn(self.norm1(value)))
        return value + self.drop_path(self.mlp(self.norm2(value)))


def orthogonal_(tensor, gain=1):
    shape = tuple(int(item) for item in tensor.shape)
    if len(shape) < 2:
        raise ValueError("orthogonal_ requires at least 2 dimensions")
    rows = shape[0]
    cols = int(np.prod(shape[1:]))
    flat = (cols, rows) if rows < cols else (rows, cols)
    array = np.random.normal(0.0, 1.0, flat)
    q, r = np.linalg.qr(array)
    signs = np.sign(np.diag(r))
    q *= signs
    if rows < cols:
        q = q.T
    q = (gain * q.reshape(shape)).astype(np.float32)
    tensor.update(jt.array(q))
    return tensor


datasets = SimpleNamespace(MNIST=MNIST)
init = SimpleNamespace(
    calculate_gain=calculate_gain,
    constant_=constant_,
    kaiming_normal_=kaiming_normal_,
    kaiming_uniform_=kaiming_uniform_,
    normal_=normal_,
    orthogonal_=orthogonal_,
    xavier_uniform_=xavier_uniform_,
)
transforms = SimpleNamespace(
    Compose=transform.Compose,
    Normalize=transform.ImageNormalize,
    ToTensor=transform.ToTensor,
)
'''


def write_lib(output: Path) -> str:
    path = output / "portlib.py"
    path.write_text(PORTLIB, encoding="utf-8")
    return path.name
