import torch
from model.feed_forward import PositionwiseFeedForward
from model.embedding import Embedding
from model.attentions import MultiHeadAttention
from model.norm import Norm
import torch.nn as nn


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
        self.attention = MultiHeadAttention(head=n_head,
                                            dim_model=dim_model,
                                            drop=drop)
        self.norm_1 = Norm(dim_model=dim_model)
        self.drop_1 = nn.Dropout(drop)

        self.feed_forward = PositionwiseFeedForward(dim_model=dim_model, hidden=feed_hidden, drop=drop)
        self.norm_2 = Norm(dim_model=dim_model)
        self.drop_2 = nn.Dropout(drop)
        self.n_layers = n_layers

    def forward(self, x, mask):
        x = self.embedding(x)

        for _ in range(self.n_layers):
            # calculate attention
            x_ = x
            x = self.attention(x ,x ,x, mask)
            self.norm_1(self.drop_1(x) + x_)

            # position feed forward
            x_ = x
            x = self.feed_forward(x)
            x = self.norm_2(self.drop_2(x) + x_)

        return x





