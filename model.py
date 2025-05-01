import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from transformers import BertConfig, BertModel

from data_preprocessing import DataPreprocessing
from data_processing import DataParameters, DataProcessing


@dataclass
class Bert4RecParams:
    vocab_size: int = 59049
    heads: int = 4
    num_hidden_layers: int = 4
    hidden_layer_size: int = 256
    num_pos: int = 20


class Bert4Rec(nn.Module):
    def __init__(self, params: Bert4RecParams):
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

    def forward(self, input_ids, attention_mask, position_ids, output_logits=True):

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)

        logits = self.output(bert_output.last_hidden_state)
        if output_logits:
            return logits
        else:
            return F.softmax(logits, -1)
