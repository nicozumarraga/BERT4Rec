import os
from turtle import forward
from requests import head
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder
from dataclasses import dataclass

@dataclass
class BertParams:
    vocab_size = 59049
    heads = 4
    layers=6,
    emb_dim=256,
    pad_id=0,
    num_pos=128


class BertModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.params = BertParams()

        self.encoder = Encoder(source_vocab_size=self.params.vocab_size,
                               emb_dim=self.params.emb_dim,
                               layers=self.params.layers,
                               heads=self.params.heads,
                               dim_model=self.params.emb_dim,
                               dim_inner=4 * self.params.emb_dim,
                               dim_value=self.params.emb_dim,
                               dim_key=self.params.emb_dim,
                               pad_id=self.params.pad_id,
                               num_pos=self.params.num_pos)

        self.reco = nn.Linear(self.params.emb_dim, self.params.vocab_size)

    def forward(self, source, source_mask):

        encoder_output = self.encoder(source, source_mask)

        output = self.reco(encoder_output)

        return output.permute(0, 2, 1)
