"""Frozen source entry for the supervised SimCSE BERT-base training GATE (v2).

Official supervised recipe: BERT-base initialization, cls pooler with the MLP in
training, temperature 0.05, NLI triplets (premise, entailment, contradiction hard
negative), AdamW lr 5e-5 with the Trainer's linear decay and no warmup. Dropout is
0 so the target can be compared without replaying dropout masks. The decay spans
the frozen GATE steps, so every step uses a different learning rate.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from simcse.models import BertForCL
from transformers import BertConfig

LR = 5e-5
TOTAL_STEPS = 16


def load_train_model() -> BertForCL:
    config = BertConfig.from_json_file(str(Path(__file__).with_name("train_config.json")))
    model_args = SimpleNamespace(
        pooler_type="cls",
        temp=0.05,
        do_mlm=False,
        mlp_only_train=False,
        hard_negative_weight=0.0,
    )
    return BertForCL(config, model_args=model_args)


def build_optimizer(model: BertForCL):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, foreach=False,
    )
    # transformers.get_linear_schedule_with_warmup(num_warmup_steps=0) 的同一公式
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, (TOTAL_STEPS - step) / TOTAL_STEPS)
    )
    return optimizer, scheduler


def train_step(model: BertForCL, state, input_ids, attention_mask, token_type_ids):
    optimizer, scheduler = state
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, attention_mask=attention_mask,
                 token_type_ids=token_type_ids, return_dict=True).loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss
