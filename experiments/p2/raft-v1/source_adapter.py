"""Controller source entry for the frozen RAFT inference and training pilots."""
from __future__ import annotations

import inspect
import os
import sys
import textwrap
from pathlib import Path

import torch

if os.environ.get("TP_RAFT_DEPS"):
    sys.path.insert(0, os.environ["TP_RAFT_DEPS"])
sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
import corr as corr_module  # noqa: E402
from corr import CorrBlock  # noqa: E402
from deterministic_sampler import official_forward_deterministic_backward  # noqa: E402
from raft import RAFT  # noqa: E402
from utils.utils import bilinear_sampler as official_bilinear_sampler  # noqa: E402
from utils.utils import coords_grid


class Args(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _args() -> Args:
    return Args(small=False, mixed_precision=False, alternate_corr=False, dropout=0)


def _configure_source() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


class SourceRAFT(RAFT):
    def initialize_flow(self, img):
        batch, _, height, width = img.shape
        grid = coords_grid(batch, height // 8, width // 8, device=img.device).to(img.dtype)
        return grid, grid.clone()

    def double(self):
        super().double()
        # Upstream explicitly casts feature maps back to FP32. This source-only
        # calibration branch keeps all float activations in FP64.
        source = textwrap.dedent(inspect.getsource(RAFT.forward))
        if source.count("fmap1 = fmap1.float()") != 1 or source.count("fmap2 = fmap2.float()") != 1:
            raise RuntimeError("upstream RAFT feature-map dtype branch changed")
        namespace = dict(RAFT.forward.__globals__)
        exec(source.replace("fmap1 = fmap1.float()", "fmap1 = fmap1.to(image1.dtype)")
                   .replace("fmap2 = fmap2.float()", "fmap2 = fmap2.to(image1.dtype)"), namespace)
        self.forward = namespace["forward"].__get__(self, type(self))
        corr_source = textwrap.dedent(inspect.getsource(CorrBlock.__call__))
        if corr_source.count(".contiguous().float()") != 1:
            raise RuntimeError("upstream correlation dtype branch changed")
        exec(corr_source.replace(".contiguous().float()", ".contiguous().to(coords.dtype)"), corr_module.__dict__)
        CorrBlock.__call__ = corr_module.__dict__["__call__"]
        return self


class InferenceRAFT(SourceRAFT):
    def forward(self, image1, image2):
        return super().forward(image1, image2, iters=20, test_mode=True)


def load_inference_model() -> InferenceRAFT:
    _configure_source()
    return InferenceRAFT(_args())


def load_train_model() -> SourceRAFT:
    _configure_source()
    corr_module.bilinear_sampler = official_forward_deterministic_backward
    return SourceRAFT(_args())


def build_optimizer(model: SourceRAFT):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=5e-5, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 2e-5, total_steps=103, pct_start=0.05,
        cycle_momentum=False, anneal_strategy="linear")
    return optimizer, scheduler


def train_step(model: SourceRAFT, state, image1, image2, flow_gt, valid):
    corr_module.bilinear_sampler = (
        official_forward_deterministic_backward
        if torch.are_deterministic_algorithms_enabled()
        else official_bilinear_sampler
    )
    optimizer, scheduler = state
    optimizer.zero_grad(set_to_none=True)
    predictions = model(image1, image2, iters=12, test_mode=False)
    magnitude = flow_gt.square().sum(dim=1).sqrt()
    mask = (valid >= 0.5) & (magnitude < 400)
    loss = sum(0.8 ** (11 - i) * (mask[:, None] * (pred - flow_gt).abs()).mean()
               for i, pred in enumerate(predictions))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    return loss
