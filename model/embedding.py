import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    def __init__(self, dim_model, max_seq_len, device) -> None:
        super().__init__()
        self.encoding = torch.zeros(max_seq_len, dim_model, device=device)
        self.encoding.requires_grad = False
        position = torch.arange(0, max_seq_len, device=device).float().unsqueeze(dim=1)
        s2i = torch.arange(0, dim_model, step=2, device=device)

        self.encoding[:, 0::2] = torch.sin(position / (10_000 ** s2i))
        self.encoding[:, 1::2] = torch.cos(position / (10_000 ** s2i))

    def forward(self, x):
        bs, seq_len = x.size()
        return self.encoding[:seq_len, :]


class TokenEmbedding(nn.Embedding):
    def __init__(self, voc_size, dim_model):
        super().__init__(voc_size, dim_model, padding_idx=1)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, dim_model):
        super().__init__(3, dim_model, padding_idx=1)


class Embedding(nn.Module):
    def __init__(self, voc_size, dim_model, max_seq_len, drop, device) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(voc_size=voc_size, dim_model=dim_model)
        self.position_embedding = PositionEmbedding(dim_model=dim_model, max_seq_len=max_seq_len, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.to(torch.int64)
        token = self.token_embedding(x)
        position = self.position_embedding(x)
        return self.drop(token + position)
