from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
import torch.nn as nn
from model.embedding import Embedding
from model.norm import Norm
from model.transformer import get_pad_mask, get_subsequent_mask



class TransformerFromTorch(nn.Module):
    def __init__(self, d_model, n_head, n_layer, src_vocab_size, trg_vocab_size, max_len, pad_idx,drop=0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.src_emb = Embedding(src_vocab_size, d_model, max_len, pad_idx, drop)
        self.trg_emb = Embedding(trg_vocab_size, d_model, max_len, pad_idx, drop)
        self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model, n_head), n_layer, norm=Norm(d_model))
        self.decoder = TransformerDecoder(TransformerDecoderLayer(d_model, n_head), n_layer, norm=Norm(d_model))

    def forward(self, src, trg):
        src_mask = get_pad_mask(src, self.pad_idx)
        trg_mask = get_pad_mask(trg, self.pad_idx) & get_subsequent_mask(trg)
        enc_out = self.encoder(self.src_emb(src), src_mask)
        dec_out = self.decoder(self.trg_emb(trg), enc_out, trg_mask, src_mask)
        return dec_out




