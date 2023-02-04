import torch
from model.feed_forward import PositionwiseFeedForward
from model.embedding import Embedding
from model.attentions import MultiHeadAttention
from model.norm import Norm
from model.connection import LayerConnection
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, n_head, feed_hidden, drop):
        super().__init__()
        self.attention = MultiHeadAttention(head=n_head, dim_model=dim_model, drop=drop)
        self.feed_forward = PositionwiseFeedForward(dim_model=dim_model, hidden=feed_hidden, drop=drop)
        self.norm = Norm(dim_model=dim_model)
        self.connection = nn.ModuleList(LayerConnection(dim_model, drop) for _ in range(2))

    def forward(self, x, mask):
        x = self.connection[0](x, lambda _x: self.attention(_x, _x, _x, mask))

        # position feed forward
        x = self.connection[1](x, self.feed_forward)
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
