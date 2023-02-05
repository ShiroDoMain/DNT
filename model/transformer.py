import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from torch.autograd import Variable

from model.models import *
import numpy as np


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(trg, pad_idx):
    """ For masking out the subsequent info. """
    trg_mask = (trg != pad_idx).unsqueeze(-2)
    trg_mask = trg_mask & Variable(
        subsequent_mask(trg.size(-1)).type_as(trg_mask.data))
    return trg_mask


class Transformer(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, pad_idx, src_embed, trg_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

    def forward(self, src, trg):
        """Take in and process masked src and target sequences."""
        src_mask = (src != self.pad_idx).unsqueeze(-2)
        trg_mask = make_std_mask(trg, self.pad_idx)
        return self.decode(self.encode(src, src_mask), src_mask,
                           trg, trg_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, trg, trg_mask):
        return self.decoder(self.trg_embed(trg), memory, src_mask, trg_mask)

    @classmethod
    def make_model(cls, src_vocab_size, trg_vocab_size, max_len, pad_idx, n_layer=6,
                   d_model=512, d_ff=2048, n_head=8, dropout=0.1):
        """Helper: Construct a model from hyperparameters."""
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionEmbedding(d_model, dropout, max_len)
        model = Transformer(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layer),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), n_layer),
            pad_idx,
            nn.Sequential(Embedding(d_model, src_vocab_size), c(position)),
            nn.Sequential(Embedding(d_model, trg_vocab_size), c(position)),
            Generator(d_model, trg_vocab_size))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
