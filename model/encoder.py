from model.embedding import Embedding
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, dim_model, drop, device):
        super().__init__()
        self.embedding = Embedding(voc_size=vocab_size,
                                   max_seq_len=max_seq_len,
                                   dim_model=dim_model,
                                   drop=drop,
                                   device=device)
