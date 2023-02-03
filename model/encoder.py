import torch
from model.feed_forward import PositionwiseFeedForward
from model.embedding import Embedding
from model.attentions import MultiHeadAttention
from model.norm import Norm
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, n_head, feed_hidden, drop):
        super().__init__()
        self.attention = MultiHeadAttention(head=n_head, dim_model=dim_model, drop=drop)
        self.feed_forward = PositionwiseFeedForward(dim_model=dim_model, hidden=feed_hidden, drop=drop)
        self.norm = Norm(dim_model=dim_model)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, mask):
        x_ = x
        x = self.attention(x, x, x, mask)
        self.norm(self.dropout(x) + x_)

        # position feed forward
        x_ = x
        x = self.feed_forward(x)
        x = self.norm(self.dropout(x) + x_)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 encoder_vocab_size,
                 max_seq_len,
                 dim_model,
                 n_layers,
                 n_head,
                 feed_hidden,
                 pad_idx,
                 drop,
                 device):
        super().__init__()
        self.embedding = Embedding(voc_size=encoder_vocab_size,
                                   max_seq_len=max_seq_len,
                                   dim_model=dim_model,
                                   drop=drop,
                                   device=device,
                                   pad_idx=pad_idx)
        self.layers = nn.ModuleList([EncoderLayer(dim_model=dim_model,
                                                  n_head=n_head,
                                                  feed_hidden=feed_hidden,
                                                  drop=drop) for _ in range(n_layers)])

    def forward(self, x, mask):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, mask)
        return x
