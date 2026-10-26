"""Fixed Jittor entry for the unsupervised SimCSE BERT-base inference pilot."""

from __future__ import annotations

import jittor as jt
from run import load_model


class TorchPortSimCSE(jt.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = load_model()

    def execute(self, input_ids, attention_mask, token_type_ids):
        chunks = []
        for start in range(0, input_ids.shape[0], 64):
            stop = start + 64
            hidden_states = self.model(input_ids[start:stop], attention_mask[start:stop],
                                       token_type_ids[start:stop])
            chunks.append(hidden_states[:, 0, :])
        return jt.concat(chunks, dim=0)
