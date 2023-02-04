import torch.nn as nn
from model.norm import Norm


class LayerConnection(nn.Module):
    def __init__(self, dim_model, drop):
        super().__init__()
        self.norm = Norm(dim_model)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

