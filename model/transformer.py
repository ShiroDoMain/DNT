from model.encoder import Encoder
from model.decoder import Decoder
import torch.nn as nn
import torch


class Transformer(nn.Module):
    def __init__(self,
                 dim_model,
                 max_seq_len,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 n_head,
                 n_layers,
                 feed_hidden,
                 pad_idx,
                 drop,
                 device):
        super().__init__()
        self.pad_idx = pad_idx
        self.device = device
        self.encoder = Encoder(
            encoder_vocab_size=encoder_vocab_size,
            max_seq_len=max_seq_len,
            dim_model=dim_model,
            n_layers=n_layers,
            n_head=n_head,
            feed_hidden=feed_hidden,
            pad_idx=pad_idx,
            drop=drop,
            device=device
        )
        self.decoder = Decoder(
            decode_vocab_size=decoder_vocab_size,
            max_seq_len=max_seq_len,
            dim_model=dim_model,
            feed_hidden=feed_hidden,
            n_layers=n_layers,
            n_head=n_head,
            pad_idx=pad_idx,
            drop=drop,
            device=device
        )

    def forward(self, source, target):
        source_mask = self.pad_mask(source, source)
        source_target_mask = self.pad_mask(target, source)

        target_mask = self.pad_mask(target, target) * self.no_peat_mask(target, target)
        encoder_source = self.encoder(source, source_mask)
        output = self.decoder(target, encoder_source, target_mask, source_target_mask)
        return output

    def pad_mask(self, q, k):
        q_len, k_len = q.size(1), k.size(1)

        k = k.ne(self.pad_idx).unsqueeze(1).unsqueeze(2).repeat(1, 1, q_len, 1)
        q = q.ne(self.pad_idx).unsqueeze(1).unsqueeze(3).repeat(1, 1, 1, k_len)

        return k & q

    def no_peat_mask(self, q, k):
        q_len, k_len = q.size(1), k.size(1)
        return torch.tril(torch.ones(q_len, k_len)).type(torch.BoolTensor).to(self.device)
