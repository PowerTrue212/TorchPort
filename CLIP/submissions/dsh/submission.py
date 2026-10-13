"""Migration submission contract — DSH baseline.

OpenAI CLIP ViT-B/32 migrated to native Jittor (no PyTorch, no TorchPort).

- `load_model(weights_path)` builds a pure `jittor.Module` and loads every
  weight from the shared FP32 `weights_fp32.npz` archive. The archive holds
  the official CLIP state_dict keys (e.g. `visual.conv1.weight`,
  `transformer.resblocks.0.attn.in_proj_weight`, `logit_scale`), all float32.
- `forward_logits(model, images, tokens)` runs the frozen fp32 forward pass on
  the prepared inputs and returns a `jittor.Var` of shape [4, 4] containing the
  image-to-text logits: `logit_scale.exp() * (L2-normalized image features) @
  (L2-normalized text features).T`.

The model mirrors the reference PyTorch implementation exactly:
  - vision: 32x32 stride-32 patch conv -> + class token -> + positional ->
    LayerNorm -> 12 x (causal-free attention + quick-GELU MLP) -> LN on CLS ->
    projection to 512-d;
  - text: embedding + positional -> 12 blocks with additive causal mask
    (-inf strictly above the diagonal) -> final LN -> EOT token feature ->
    projection to 512-d;
  - LayerNorm eps=1e-5, affine; MultiheadAttention uses the additive mask,
    no dropout (eval mode).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import jittor as jt


class LayerNorm(jt.Module):
    """Equivalent of torch.nn.LayerNorm(dim) (eps=1e-5, affine) in float32."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = jt.ones((dim,))
        self.bias = jt.zeros((dim,))

    def execute(self, x: jt.Var) -> jt.Var:
        mean = x.mean(dim=-1, keepdims=True)
        var = ((x - mean) * (x - mean)).mean(dim=-1, keepdims=True)
        inv_std = 1.0 / jt.sqrt(var + self.eps)
        return (x - mean) * inv_std * self.weight + self.bias


class Linear(jt.Module):
    """Plain affine layer with torch-style weight [out, in] / bias [out]."""

    def __init__(self, out_features: int, in_features: int):
        self.weight = jt.zeros((out_features, in_features))
        self.bias = jt.zeros((out_features,))

    def execute(self, x: jt.Var) -> jt.Var:
        return jt.matmul(x, self.weight.transpose(0, 1)) + self.bias


class QuickGELU(jt.Module):
    def execute(self, x: jt.Var) -> jt.Var:
        return x * jt.sigmoid(1.702 * x)


class MultiheadAttention(jt.Module):
    """Equivalent of torch.nn.MultiheadAttention(d_model, n_head) called as
    attn(x, x, x, need_weights=False, attn_mask=mask) on a [L, N, E] input,
    in eval mode (dropout disabled), with an additive float mask."""

    def __init__(self, d_model: int, n_head: int, attn_mask: Optional[np.ndarray] = None):
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.in_proj_weight = jt.zeros((3 * d_model, d_model))
        self.in_proj_bias = jt.zeros((3 * d_model,))
        self.out_proj = Linear(d_model, d_model)
        if attn_mask is not None:
            self.attn_mask = jt.array(np.ascontiguousarray(attn_mask, dtype=np.float32))
        else:
            self.attn_mask = None

    def execute(self, x: jt.Var) -> jt.Var:
        # x: [L, N, E]
        L, N, E = x.shape
        hd = self.head_dim
        qkv = jt.matmul(x, self.in_proj_weight.transpose(0, 1)) + self.in_proj_bias
        q = qkv[:, :, :E].reshape(L, N * self.n_head, hd).transpose(0, 1)
        k = qkv[:, :, E:2 * E].reshape(L, N * self.n_head, hd).transpose(0, 1)
        v = qkv[:, :, 2 * E:].reshape(L, N * self.n_head, hd).transpose(0, 1)
        q = q * (hd ** -0.5)
        attn = jt.matmul(q, k.transpose(1, 2))  # [N*H, L, L]
        if self.attn_mask is not None:
            attn = attn + self.attn_mask  # broadcast [L, L] over [N*H, L, L]
        attn = jt.nn.softmax(attn, dim=-1)
        out = jt.matmul(attn, v)  # [N*H, L, hd]
        out = out.transpose(0, 1).reshape(L, N, E)
        return self.out_proj(out)


class MLP(jt.Module):
    def __init__(self, d_model: int):
        self.c_fc = Linear(4 * d_model, d_model)
        self.c_proj = Linear(d_model, 4 * d_model)

    def execute(self, x: jt.Var) -> jt.Var:
        h = self.c_fc(x)
        h = h * jt.sigmoid(1.702 * h)
        return self.c_proj(h)


class ResidualAttentionBlock(jt.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: Optional[np.ndarray] = None):
        self.attn = MultiheadAttention(d_model, n_head, attn_mask)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = MLP(d_model)
        self.ln_2 = LayerNorm(d_model)

    def execute(self, x: jt.Var) -> jt.Var:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(jt.Module):
    def __init__(self, width: int, layers: int, heads: int,
                 attn_mask: Optional[np.ndarray] = None):
        self.width = width
        self.layers = layers
        self.resblocks = [
            ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)
        ]

    def execute(self, x: jt.Var) -> jt.Var:
        for block in self.resblocks:
            x = block(x)
        return x


class VisionTransformer(jt.Module):
    def __init__(self, input_resolution: int = 224, patch_size: int = 32,
                 width: int = 768, layers: int = 12, heads: int = 12,
                 output_dim: int = 512):
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = jt.nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        self.class_embedding = jt.zeros((width,))
        self.positional_embedding = jt.zeros(
            ((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, attn_mask=None)
        self.ln_post = LayerNorm(width)
        self.proj = jt.zeros((width, output_dim))

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.conv1(x)  # [N, width, grid, grid]
        N = x.shape[0]
        C = x.shape[1]
        x = x.reshape(N, C, -1).permute(0, 2, 1)  # [N, grid^2, width]
        x = jt.concat([self.class_embedding.broadcast((N, 1, C)), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        x = jt.matmul(x, self.proj)
        return x


class CLIP(jt.Module):
    """Jittor-native CLIP ViT-B/32 with the official state_dict layout."""

    def __init__(self, weights: Dict[str, np.ndarray]):
        # Vision tower (ViT-B/32: 224px, 32px patches, 12 x 768-d, 12 heads).
        self.visual = VisionTransformer(
            input_resolution=224, patch_size=32, width=768, layers=12,
            heads=12, output_dim=512,
        )
        # Text tower (12 x 512-d, 8 heads, causal additive mask).
        causal_mask = np.full((77, 77), -np.inf, dtype=np.float32)
        causal_mask = np.triu(causal_mask, 1)  # -inf above diagonal, 0 elsewhere
        self.transformer = Transformer(width=512, layers=12, heads=8,
                                       attn_mask=causal_mask)
        self.token_embedding = jt.nn.Embedding(49408, 512)
        self.positional_embedding = jt.zeros((77, 512))
        self.ln_final = LayerNorm(512)
        self.text_projection = jt.zeros((512, 512))
        self.logit_scale = jt.zeros(())
        self._load_weights(weights)

    def _load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        for key, array in weights.items():
            parts = key.split(".")
            obj: Any = self
            for part in parts[:-1]:
                if part == "resblocks":
                    obj = obj.resblocks
                elif part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            value = jt.array(np.ascontiguousarray(array, dtype=np.float32))
            setattr(obj, parts[-1], value)

    def encode_image(self, image: jt.Var) -> jt.Var:
        return self.visual(image)

    def encode_text(self, text: jt.Var, eot_mask: jt.Var) -> jt.Var:
        x = self.token_embedding(text)  # [N, L, D]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # take features from the EOT (highest token id) embedding
        x = (x * eot_mask).sum(dim=1)  # [N, D]
        x = jt.matmul(x, self.text_projection)
        return x

    def execute(self, image: jt.Var, text: jt.Var, eot_mask: jt.Var) -> jt.Var:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, eot_mask)

        # normalized features
        image_features = image_features / jt.sqrt(
            (image_features * image_features).sum(dim=1, keepdims=True)
        )
        text_features = text_features / jt.sqrt(
            (text_features * text_features).sum(dim=1, keepdims=True)
        )

        # cosine similarity as logits
        logit_scale = jt.exp(self.logit_scale)
        logits_per_image = logit_scale * jt.matmul(
            image_features, text_features.transpose(0, 1)
        )
        return logits_per_image


def load_model(weights_path: Path) -> Any:
    """Build a jittor.Module and load weights from the shared FP32 NPZ archive."""
    archive = np.load(str(weights_path))
    weights = {key: archive[key] for key in archive.files}
    return CLIP(weights)


def forward_logits(model: Any, images: np.ndarray, tokens: np.ndarray) -> Any:
    """Return one jittor.Var with shape [4, 4] (image-to-text CLIP logits)."""
    images_array = np.ascontiguousarray(images, dtype=np.float32)
    tokens_array = np.ascontiguousarray(tokens, dtype=np.int32)
    batch, length = tokens_array.shape

    # EOT position per row == argmax of token ids (EOT id 49407 is the max).
    eot_positions = np.argmax(tokens_array, axis=1)
    selection = np.zeros((batch, length, 1), dtype=np.float32)
    selection[np.arange(batch), eot_positions, 0] = 1.0

    image_var = jt.array(images_array)
    token_var = jt.array(tokens_array)
    eot_mask = jt.array(selection)
    return model(image_var, token_var, eot_mask)
