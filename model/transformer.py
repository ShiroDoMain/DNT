import torch
import torch.nn as nn
from torch.autograd import Variable

from model.decoder import Decoder
from model.encoder import Encoder


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
        self.dim_model = dim_model
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
        source_mask = (source != self.pad_idx).unsqueeze(-2)

        target_mask = self.make_std_mask(target)
        encoder_source = self.encoder(source, source_mask)
        output = self.decoder(target, encoder_source, target_mask, source_mask)
        return output

    def make_std_mask(self, tgt):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != self.pad_idx).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            self.subsequent_mask(tgt.size(-1))).to(self.device)
        return tgt_mask

    def subsequent_mask(self, size):
        return torch.ones((1, size, size)).triu(1) == 0
