import torch
import torch.nn as nn
from model.embedding import Embedding
from model.attentions import MultiHeadAttention
from model.norm import Norm
from model.feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, dim_model, n_head, feed_hidden, drop):
        super().__init__()
        self.self_attention = MultiHeadAttention(head=n_head, dim_model=dim_model, drop=drop)

        self.encoder_decoder_attention = MultiHeadAttention(dim_model=dim_model, head=n_head, drop=drop)

        self.feed_forward = PositionwiseFeedForward(dim_model=dim_model, hidden=feed_hidden, drop=drop)
        self.norm = Norm(dim_model=dim_model)
        self.dropout = nn.Dropout(drop)

    def forward(self, target, encode_source, target_mask, source_mask):
        target_ = target
        target = self.self_attention(target, target, target, mask=target_mask)
        target = self.norm(self.dropout(target) + target_)

        if encode_source is not None:
            target_ = target
            target = self.encoder_decoder_attention(target, encode_source, encode_source, source_mask)

            target = self.norm(self.dropout(target) + target_)

        target_ = target
        target = self.feed_forward(target)
        return self.norm(self.dropout(target) + target_)


class Decoder(nn.Module):
    def __init__(self, decode_vocab_size, max_seq_len, dim_model, feed_hidden, pad_idx, n_layers, n_head, drop, device):
        super().__init__()
        self.embedding = Embedding(voc_size=decode_vocab_size,
                                   dim_model=dim_model,
                                   max_seq_len=max_seq_len,
                                   pad_idx=pad_idx,
                                   drop=drop,
                                   device=device)
        self.layers = nn.ModuleList(DecoderLayer(dim_model, n_head, feed_hidden, drop) for _ in range(n_layers))

        self.linear = nn.Linear(dim_model, decode_vocab_size)

    def forward(self, target, encode_source, target_mask, source_mask):
        target = self.embedding(target)

        for layer in self.layers:
            target = layer(target, encode_source, target_mask, source_mask)

        return self.linear(target)
