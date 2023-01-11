import torch.nn as nn
import torch
import math


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):

    def __init__(self, dim_model, hidden, drop):
        super().__init__()
        self.linear1 = nn.Linear(dim_model, hidden)
        self.linear2 = nn.Linear(hidden, dim_model)
        self.gelu = GELU()
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))

