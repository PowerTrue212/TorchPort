from __future__ import annotations

import jittor as jt
from jittor import nn


class TorchLayerNorm(nn.Module):
    """使用中心化方差公式的 LayerNorm，匹配 PyTorch 的数值语义。"""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = jt.ones((normalized_shape,), dtype="float32")
        self.bias = jt.zeros((normalized_shape,), dtype="float32")

    def execute(self, value: jt.Var) -> jt.Var:
        mean = value.mean(dim=-1, keepdims=True)
        centered = value - mean
        variance = (centered * centered).mean(dim=-1, keepdims=True)
        normalized = centered / jt.sqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class GELU(nn.Module):
    def execute(self, value: jt.Var) -> jt.Var:
        return nn.gelu(value)


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int = 192) -> None:
        self.proj = nn.Conv(
            3,
            embed_dim,
            kernel_size=16,
            stride=16,
            padding=0,
            bias=True,
        )

    def execute(self, value: jt.Var) -> jt.Var:
        value = self.proj(value)
        return value.reshape((value.shape[0], value.shape[1], -1)).transpose(0, 2, 1)


class Attention(nn.Module):
    def __init__(self, dim: int = 192, num_heads: int = 3) -> None:
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def execute(self, value: jt.Var) -> jt.Var:
        batch, tokens, channels = value.shape
        qkv = self.qkv(value).reshape(
            (batch, tokens, 3, self.num_heads, self.head_dim)
        )
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        query, key, item_value = qkv[0], qkv[1], qkv[2]
        scores = jt.matmul(query * self.scale, key.transpose(0, 1, 3, 2))
        probabilities = scores.softmax(dim=-1)
        context = jt.matmul(probabilities, item_value)
        context = context.transpose(0, 2, 1, 3).reshape((batch, tokens, channels))
        return self.proj(context)


class Mlp(nn.Module):
    def __init__(self, dim: int = 192, hidden_dim: int = 768) -> None:
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.act = GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def execute(self, value: jt.Var) -> jt.Var:
        return self.fc2(self.act(self.fc1(value)))


class Block(nn.Module):
    def __init__(self, dim: int = 192, num_heads: int = 3) -> None:
        self.norm1 = TorchLayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = TorchLayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * 4)

    def execute(self, value: jt.Var) -> jt.Var:
        value = value + self.attn(self.norm1(value))
        return value + self.mlp(self.norm2(value))


class VisionTransformerTiny(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        embed_dim = 192
        self.patch_embed = PatchEmbed(embed_dim)
        self.cls_token = jt.zeros((1, 1, embed_dim), dtype="float32")
        self.pos_embed = jt.zeros((1, 197, embed_dim), dtype="float32")
        self.blocks = nn.Sequential(*[Block(embed_dim, 3) for _ in range(12)])
        self.norm = TorchLayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes, bias=True)

    def embed(self, value: jt.Var) -> jt.Var:
        value = self.patch_embed(value)
        class_token = self.cls_token.repeat(value.shape[0], 1, 1)
        value = jt.concat((class_token, value), dim=1)
        return value + self.pos_embed

    def execute(self, value: jt.Var) -> jt.Var:
        value = self.embed(value)
        value = self.blocks(value)
        value = self.norm(value)
        return self.head(value[:, 0])
