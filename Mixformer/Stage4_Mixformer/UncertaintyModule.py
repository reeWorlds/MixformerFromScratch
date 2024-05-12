import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from . import Mixformer


def make_uncertainty_config(size_type='medium'):
    if size_type == 'small':
        embd_d = 48
    elif size_type == 'medium':
        embd_d = 72
    elif size_type == 'large':
        embd_d = 108
    else:
        raise ValueError(f'Invalid size type {size_type}')
    config = {'embd_d': embd_d}
    return config


class _MyAttention(nn.Module):
    def __init__(self, embd_d):
        super().__init__()
        self.embd_d = embd_d
        self.scale = embd_d ** -0.5

        self.q = nn.Linear(embd_d, embd_d)
        self.k = nn.Linear(embd_d, embd_d)
        self.v = nn.Linear(embd_d, embd_d)

    def forward(self, cls, sequence):
        q = self.q(cls)
        k = self.k(sequence)
        v = self.v(sequence)

        attn = torch.einsum('bid,bjd->bij', [q, k]) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bij,bjd->bid', attn, v)

        return out


class _MyFeedForward(nn.Module):
    def __init__(self, embd_d):
        super().__init__()
        self.embd_d = embd_d

        self.net = nn.Sequential(
            nn.Linear(embd_d, embd_d * 2),
            nn.GELU(),
            nn.Linear(embd_d * 2, embd_d)
        )

    def forward(self, x):
        return self.net(x)


class UncertaintyModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embd_d = config['embd_d']

        self.register_parameter("cls", nn.Parameter(torch.randn(1, 1, self.embd_d)))
        self.attn1 = _MyAttention(self.embd_d)
        self.ff1 = _MyFeedForward(self.embd_d)
        self.attn2 = _MyAttention(self.embd_d)
        self.ff2 = _MyFeedForward(self.embd_d)
        self.norm = nn.LayerNorm(self.embd_d)
        self.linear1 = nn.Linear(self.embd_d, self.embd_d)
        self.linear2 = nn.Linear(self.embd_d, 1)

    def forward(self, search, target):
        B, N, D = search.shape

        x = self.cls.expand(B, -1, -1)
        x = x + self.attn1(x, search)
        x = x + self.ff1(x)
        x = x + self.attn2(x, target)
        x = x + self.ff2(x)
        x = self.norm(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x
