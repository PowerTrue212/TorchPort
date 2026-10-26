"""Frozen source entry for the unsupervised SimCSE BERT-base inference pilot."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import BertConfig, BertModel


class TorchPortSimCSE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = BertConfig.from_json_file(str(Path(__file__).with_name("task_config.json")))
        self.model = BertModel(config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        chunks = []
        for start in range(0, input_ids.shape[0], 64):
            stop = start + 64
            output = self.model(input_ids[start:stop], attention_mask[start:stop],
                                token_type_ids=token_type_ids[start:stop],
                                return_dict=True)
            chunks.append(output.last_hidden_state[:, 0, :])
        return torch.cat(chunks, dim=0)
