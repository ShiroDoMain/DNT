import torch
import  torch.nn as nn
from model.embedding import Embedding
from model.attentions import MultiHeadAttention
from model.norm import Norm
from model.feed_forward import PositionwiseFeedForward


class Decoder(nn.Module):
    def __init__(self, decode_vocab_size, max_seq_len, dim_model, feed_hidden, pad_idx, n_layers, n_head, drop, device):
        super().__init__()
        self.embedding = Embedding(voc_size=decode_vocab_size,
                                   dim_model=dim_model,
                                   max_seq_len=max_seq_len,
                                   pad_idx=pad_idx,
                                   drop=drop,
                                   device=device)
        self.layers = n_layers

        self.self_attention = MultiHeadAttention(head=n_head, dim_model=dim_model, drop=drop)
        self.norm_1 = Norm(dim_model=dim_model)
        self.drop_1 = nn.Dropout(drop)

        self.encoder_decoder_attention = MultiHeadAttention(dim_model=dim_model, head=n_head, drop=drop)
        self.norm_2 = Norm(dim_model=dim_model)
        self.drop_2 = nn.Dropout(drop)

        self.feed_forward = PositionwiseFeedForward(dim_model=dim_model, hidden=feed_hidden, drop=drop)
        self.norm_3 = Norm(dim_model=dim_model)
        self.drop_3 = nn.Dropout(drop)

        self.linear = nn.Linear(dim_model, decode_vocab_size)

    def forward(self, target, encode_source, mask, source_mask):
        target = self.embedding(target)

        for _ in range(self.layers):
            target_ = target
            target = self.self_attention(target, target, target, mask=mask)
            target = self.norm_1(self.drop_1(target) + target_)

            if encode_source is not None:
                target_ = target
                target = self.encoder_decoder_attention(target, target, target, source_mask)

                target = self.norm_2(self.drop_2(target) + target_)

            target_ = target
            target = self.feed_forward(target)
            target = self.norm_3(self.drop_3(target) + target_)

        return self.linear(target)








