"""Frozen source entry for the SimCSE BERT-base diagnostic training GATE."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from simcse.models import BertForCL
from transformers import BertConfig


def load_train_model() -> BertForCL:
    config = BertConfig.from_json_file(str(Path(__file__).with_name("train_config.json")))
    model_args = SimpleNamespace(
        pooler_type="cls",
        temp=0.05,
        do_mlm=False,
        mlp_only_train=True,
        hard_negative_weight=0.0,
    )
    return BertForCL(config, model_args=model_args)


def build_optimizer(model: BertForCL) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(), lr=3e-5, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, foreach=False,
    )


def train_step(model: BertForCL, optimizer: torch.optim.AdamW,
               input_ids, attention_mask, token_type_ids):
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, attention_mask=attention_mask,
                 token_type_ids=token_type_ids, return_dict=True).loss
    loss.backward()
    optimizer.step()
    return loss
