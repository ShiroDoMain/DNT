import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionEmbedding(nn.Module):
    def __init__(self, dim_model, max_seq_len) -> None:
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_len, dim_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) *
                             -(math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return Variable(self.pe[:, :x.size(1)], requires_grad=False)


class TokenEmbedding(nn.Embedding):
    def __init__(self, voc_size, dim_model, pad_idx):
        super().__init__(voc_size, dim_model, padding_idx=pad_idx)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, dim_model, pad_idx):
        super().__init__(3, dim_model, padding_idx=pad_idx)


class Embedding(nn.Module):
    def __init__(self, voc_size, dim_model, max_seq_len, pad_idx, drop) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(voc_size=voc_size, dim_model=dim_model, pad_idx=pad_idx)
        self.position_embedding = PositionEmbedding(dim_model=dim_model, max_seq_len=max_seq_len)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        token = self.token_embedding(x)
        position = self.position_embedding(x)
        return self.drop(token + position)
