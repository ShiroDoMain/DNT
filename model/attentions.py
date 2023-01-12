import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.modules import ModuleList


class Attention(nn.Module):
    def forward(self, q, k, v, mask=None, drop_fn=None, e=-1e-9):
        score = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        if mask is not None:
            score = score.masked_fill(mask == 0, e)
        attn = F.softmax(score, dim=-1)
        if drop_fn is not None:
            attn = drop_fn(attn)
        return attn @ v, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, head, dim_model, drop):
        super().__init__()
        assert dim_model % head == 0, "head mod dim must eq 0"
        self.d_k = dim_model // head
        self.head = head

        self.linear_layers = ModuleList(nn.Linear(dim_model, dim_model) for _ in range(3))
        self.output_linear = nn.Linear(dim_model, dim_model)
        self.attention = Attention()

        self.drop = nn.Dropout(drop)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q, k, v = [linear(x).view(bs, -1, self.head, self.d_k).transpose(1, 2) for linear, x in
                   zip(self.linear_layers, (q, k, v))]

        x, attn = self.attention(q, k, v, mask, self.drop)

        x = x.transpose(1, 2).contiguous().view(bs, -1, self.head * self.d_k)
        return self.output_linear(x)
