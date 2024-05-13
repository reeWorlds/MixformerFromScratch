import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from .Mixformer import MixFormer


def extract_MixFormer_features(mixformer: MixFormer, search, target):
    with torch.no_grad():
        search = rearrange(search, 'b h w c -> b c h w').contiguous()
        search = F.selu(mixformer.init_projection(search))
        search = rearrange(search, 'b c h w -> b (h w) c').contiguous()

        target = rearrange(target, 'b h w c -> b c h w').contiguous()
        target = F.selu(mixformer.init_projection(target))
        target = rearrange(target, 'b c h w -> b (h w) c').contiguous()

        x = torch.cat([search, target], dim=1)
        for stage in mixformer.stages:
            x = stage(x)

        SHW2 = mixformer.search_out_hw * mixformer.search_out_hw
        THW2 = mixformer.target_out_hw * mixformer.target_out_hw

        cls, search, target = torch.split(x, [1, SHW2, THW2], dim=1)
        # (B, D)
        cls = cls.squeeze(1)
        
    return cls, search, target


def make_uncertainty_config(size_type='medium'):
    if size_type == 'small':
        embd_d = 48
    elif size_type == 'medium':
        embd_d = 72
    elif size_type == 'large':
        embd_d = 108
    else:
        raise ValueError(f'Invalid size type {size_type}')
    config = {'embd_d': embd_d, 'target_hw': 12, 'search_hw': 16, 'num_blocks': 4}
    return config


class _MyAttentionBlock(nn.Module):
    def __init__(self, embd_d):
        super().__init__()
        self.embd_d = embd_d
        self.scale = embd_d ** -0.5

        self.norm1 = nn.LayerNorm(embd_d)
        self.q = nn.Linear(embd_d, embd_d)
        self.k = nn.Linear(embd_d, embd_d)
        self.v = nn.Linear(embd_d, embd_d)

        self.norm2 = nn.LayerNorm(embd_d)
        self.ff = nn.Sequential(
            nn.Linear(embd_d, embd_d * 2),
            nn.GELU(),
            nn.Linear(embd_d * 2, embd_d)
        )

    def forward(self, x):
        y = self.norm1(x)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        attn = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = torch.einsum('bij,bjd->bid', attn, v)

        x = x + attn
        x = x + self.ff(self.norm2(x))

        return x


class UncertaintyModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embd_d = config['embd_d']
        self.target_hw = config['target_hw']
        self.search_hw = config['search_hw']
        self.num_blocks = config['num_blocks']

        self.register_parameter("cls", nn.Parameter(torch.randn(1, 1, self.embd_d)))
        self.register_parameter("search", nn.Parameter(torch.randn(1, self.search_hw * self.search_hw, self.embd_d) * 0.5))
        self.register_parameter("target", nn.Parameter(torch.randn(1, self.target_hw * self.target_hw, self.embd_d) * 0.5))

        self.blocks = nn.ModuleList([_MyAttentionBlock(self.embd_d) for _ in range(self.num_blocks)])
        self.norm = nn.LayerNorm(self.embd_d)
        self.linear1 = nn.Linear(self.embd_d, self.embd_d)
        self.linear2 = nn.Linear(self.embd_d, 1)

    def forward(self, search, target):
        B, Ns, D = search.shape
        B, Nt, D = target.shape

        cls_exp = self.cls.expand(B, -1, -1)
        self_exp = search + self.search.expand(B, -1, -1)
        target_exp = target + self.target.expand(B, -1, -1)

        x = torch.cat([cls_exp, self_exp, target_exp], dim=1)

        for block in self.blocks:
            x = block(x)

        x, rest = torch.split(x, [1, Ns + Nt], dim=1)
        
        x = x.squeeze(1)
        x = self.norm(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x
