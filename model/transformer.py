import torch
import torch.nn as nn
from torch.autograd import Variable

from model.decoder import Decoder
from model.encoder import Encoder


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

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
        source_mask = get_pad_mask(source, self.pad_idx)
        target_mask = get_pad_mask(target, self.pad_idx) & get_subsequent_mask(target)
        encoder_output = self.encoder(source, source_mask)
        output = self.decoder(target, encoder_output, target_mask, source_mask)
        return output

