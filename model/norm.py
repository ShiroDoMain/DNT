from torch.nn import Parameter
import torch.nn as nn
import torch


class Norm(nn.Module):
    def __init__(self, dim_model, eps=1e-12):
        super().__init__()
        self.gamma = Parameter(torch.ones(dim_model))
        self.beta = Parameter(torch.zeros(dim_model))
        self.eps = eps

    def forward(self,x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta