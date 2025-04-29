import os
from turtle import forward
from typing import Optional, Tuple
from requests import head
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder
from dataclasses import dataclass

from transformers import BertConfig, BertModel, BertTokenizer


@dataclass
class Bert4RecParams:
    vocab_size: int = 59049
    heads: int = 4
    num_hidden_layers: int = 4  # TODO: use params from bert4rec
    hidden_layer_size: int = 256  # TODO: use params from bert4rec
    emb_dim: Tuple[int, ...] = (256,)  # TODO: implement
    num_pos = 128
    pad_id: int = 0
    mask_id: int = 59050


class Bert4Rec(nn.Module):
    def __init__(self, params: Optional[Bert4RecParams]):
        super().__init__()
        self.params = params

        self.bert_config = BertConfig(
            hidden_act="gelu",  # Hardcode gelu because bert4rec paper defines this
            vocab_size=params.vocab_size,
            num_attention_heads=params.heads,
            num_hidden_layers=params.num_hidden_layers,
            hidden_size=params.hidden_layer_size,
            max_position_embeddings=params.num_pos,
        )
        self.bert: BertModel = BertModel(self.bert_config)
        self.output = nn.Linear(params.hidden_layer_size, params.vocab_size)

    # TODO: further implement this with new data loader
    def forward(self, source, source_mask):

        bert_output = self.bert(source, source_mask)
        bert_output.last_hidden_state
        encoder_output = self.output(bert_output.last_hidden_state)

        output = self.reco(encoder_output)

        return output.permute(0, 2, 1)
