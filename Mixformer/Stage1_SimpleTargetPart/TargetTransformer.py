import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath


class StagePreprocessor(nn.Module):
    """
    Preprocess the search and target images to be used by attention mechanism.

    Input shape: (B, Ht * Wt, C)
    Output shape: (B, N, D); N = _Ht * _Wt
    """
    def __init__(self, config):
        super(StagePreprocessor, self).__init__()

        self.channels = config['channels']
        self.embed_dim = config['embed_dim']
        self.target_inp_h = config['target_inp_h']
        self.target_inp_w = config['target_inp_w']
        self.target_out_h = config['target_out_h']
        self.target_out_w = config['target_out_w']
        self.patch_size = config['patch_size']
        self.patch_stride = config['patch_stride']
        self.patch_padding = config['patch_padding']

        self.proj = nn.Conv2d(self.channels, self.embed_dim, self.patch_size, self.patch_stride, self.patch_padding)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        B = x.shape[0]

        # x: (B, Ht * Wt, C)
        assert x.shape == (B, self.target_inp_h * self.target_inp_w, self.channels)

        # (B, C, Ht, Wt)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.target_inp_h, w=self.target_inp_w).contiguous()
        # (B, D, _Ht, _Wt)
        x = self.proj(x)
        # x: (B, D, _Ht, _Wt)
        assert x.shape == (B, self.embed_dim, self.target_out_h, self.target_out_w)
        # (B, _Ht * _Wt, D)
        x = rearrange(x, 'b d h w -> b (h w) d').contiguous()
        # (B, _Ht * _Wt, D)
        x = self.norm(x)

        # (B, N, D)
        return x


class DepthWiseQueryKeyValue(nn.Module):
    """
    Depth-wise CNN + query, key, value projection.

    Input shape: (B, N, C); N = Ht * Wt
    Output shape: (B, H, _Nt, C/H), (B, H, __Nt, C/H), (B, H, __Nt, C/H)
    """
    def __init__(self, config):
        super(DepthWiseQueryKeyValue, self).__init__()
        assert config['embed_dim'] % config['num_heads'] == 0
        
        self.embed_dim = config['embed_dim']
        self.target_inp_h = config['target_inp_h']
        self.target_inp_w = config['target_inp_w']
        self.target_q_h = config['target_q_h']
        self.target_q_w = config['target_q_w']
        self.target_kv_h = config['target_kv_h']
        self.target_kv_w = config['target_kv_w']
        self.kernel_size = config['kernel_size']
        self.padding_q = config['padding_q']
        self.stride_q = config['stride_q']
        self.padding_kv = config['padding_kv']
        self.stride_kv = config['stride_kv']
        self.num_heads = config['num_heads']
        self.head_dim = self.embed_dim // self.num_heads

        self.norm1 = nn.LayerNorm(self.embed_dim)

        self.depthwise_q = self.build_conv_proj(self.embed_dim, self.kernel_size, self.padding_q, self.stride_q)
        self.depthwise_k = self.build_conv_proj(self.embed_dim, self.kernel_size, self.padding_kv, self.stride_kv)
        self.depthwise_v = self.build_conv_proj(self.embed_dim, self.kernel_size, self.padding_kv, self.stride_kv)

        self.proj_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_v = nn.Linear(self.embed_dim, self.embed_dim)

    def build_conv_proj(self, channels, kernel_size, padding, stride):
        proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(channels),
            Rearrange('b c h w -> b (h w) c')
        )
        return proj
    
    def forward(self, x):
        B = x.shape[0]
        Nt = self.target_inp_h * self.target_inp_w

        # (B, N, C)
        x = self.norm1(x)

        # x: (B, Nt, C)
        assert x.shape == (B, Nt, self.embed_dim)
        
        # (B, C, Ht, Wt)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.target_inp_h, w=self.target_inp_w).contiguous()
        # (B, _Nt, C)
        target_q = self.depthwise_q(x)
        # (B, _Nt, C)
        assert target_q.shape == (B, self.target_q_h * self.target_q_w, self.embed_dim)
        # (B, __Nt, C)
        target_k = self.depthwise_k(x)
        # (B, __Nt, C)
        assert target_k.shape == (B, self.target_kv_h * self.target_kv_w, self.embed_dim)
        # (B, __Nt, C)
        target_v = self.depthwise_v(x)
        # (B, __Nt, C)
        assert target_v.shape == (B, self.target_kv_h * self.target_kv_w, self.embed_dim)
        # (B, _Nt, C)
        target_q = self.proj_q(target_q)
        # (B, H, _Nt, C/H)
        target_q = rearrange(target_q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim).contiguous()
        # (B, __Nt, C)
        target_k = self.proj_k(target_k)
        # (B, H, __Nt, C/H)
        target_k = rearrange(target_k, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim).contiguous()
        # (B, __Nt, C)
        target_v = self.proj_v(target_v)
        # (B, H, __Nt, C/H)
        target_v = rearrange(target_v, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim).contiguous()
        
        # (B, H, _Nt, D/H), (B, H, __Nt, D/H), (B, H, __Nt, D/H)
        return target_q, target_k, target_v


class MultiHeadAttention(nn.Module):
    """
    Asymetric Multi-Head Attention Described in the paper.
    
    Input shape: (B, N, D), (B, H, _Nt, D/H), (B, H, __Nt, D/H), (B, H, __Nt, D/H)
    Output shape: (B, N, D)
    """
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        assert config['embed_dim'] % config['num_heads'] == 0

        self.embed_dim = config['embed_dim']
        self.target_inp_h = config['target_inp_h']
        self.target_inp_w = config['target_inp_w']
        self.target_q_h = config['target_q_h']
        self.target_q_w = config['target_q_w']
        self.target_kv_h = config['target_kv_h']
        self.target_kv_w = config['target_kv_w']
        self.num_heads = config['num_heads']
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1 / (self.head_dim ** 0.5)
        self.ff_scale = config['ff_scale']

        self.drop1 = DropPath(0.2)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.drop2 = DropPath(0.2)
        self.ff_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * self.ff_scale),
            nn.GELU(),
            nn.Linear(self.embed_dim * self.ff_scale, self.embed_dim)
        )

    def forward(self, x, target_q, target_k, target_v):
        B = target_q.shape[0]

        # target_q: (B, H, _Nt, D/H)
        assert target_q.shape == (B, self.num_heads, self.target_q_h * self.target_q_w, self.head_dim)
        # target_k: (B, H, __Nt, D/H)
        assert target_k.shape == (B, self.num_heads, self.target_kv_h * self.target_kv_w, self.head_dim)
        # target_v: (B, H, __Nt, D/H)
        assert target_v.shape == (B, self.num_heads, self.target_kv_h * self.target_kv_w, self.head_dim)

        # (B, H, _Nt, __Nt)
        target_attn = torch.einsum('bhnd,bhmd->bhnm', [target_q, target_k]) * self.scale
        # (B, H, _Nt, __Nt)
        target_attn = F.softmax(target_attn, dim=-1)
        # (B, H, _Nt, D/H)
        target_attn = torch.einsum('bhnm,bhmd->bhnd', [target_attn, target_v])
        # (B, _Nt, D)
        target_attn = rearrange(target_attn, 'b h n d -> b n (h d)').contiguous()
        # (B, _Nt, D)
        assert target_attn.shape == (B, self.target_inp_h * self.target_inp_w, self.embed_dim)
        
        # (B, [1 +] _Ns + _Nt, D)
        x = x + self.drop1(target_attn)
        # (B, [1 +] _Ns + _Nt, D)
        x = x + self.drop1(self.ff_proj(self.norm2(x)))

        # (B, N, D)
        return x


class MixedAttentionModule(nn.Module):
    """
    Mixed Attention Module described in the paper.

    Input shape: (B, N, D)
    Output shape: (B, N, D)
    """
    def __init__(self, config):
        super(MixedAttentionModule, self).__init__()

        self.embed_dim = config['embed_dim']

        self.depthwise_qkv = DepthWiseQueryKeyValue(config['depthwise_qkv'])
        self.attention = MultiHeadAttention(config['attention'])

    def forward(self, x):
        # (B, H, _Nt, D/H), (B, H, __Nt, D/H), (B, H, __Nt, D/H)
        target_q, target_k, target_v = self.depthwise_qkv(x)
        # (B, N, D)
        x = self.attention(x, target_q, target_k, target_v)

        # (B, N, D)
        return x


class Stage(nn.Module):
    """
    One stage of the ConvolutionalVisionTransformer (CVT) model.
    
    Input shape: (B, Ht * Wt, C)
    Output shape: (B, _Ht * _Wt, D)
    """
    def __init__(self, config):
        super(Stage, self).__init__()
        
        self.channels = config['channels']
        self.embed_dim = config['embed_dim']
        self.target_inp_h = config['target_inp_h']
        self.target_inp_w = config['target_inp_w']
        self.target_out_h = config['target_out_h']
        self.target_out_w = config['target_out_w']
        self.num_mam_blocks = config['num_mam_blocks']

        target_embd = self.get_pos_embd(self.target_out_h * self.target_out_w, self.embed_dim)
        self.register_buffer('positional_embd', target_embd)
        self.preprocessor = StagePreprocessor(config['preprocessor'])
        self.mam_blocks = nn.ModuleList([MixedAttentionModule(config['mam']) for _ in range(self.num_mam_blocks)])

    def get_pos_embd(self, n, d):
        assert d % 2 == 0
        scale_coef = 0.5
        
        # (N, 1)
        position = torch.arange(n).unsqueeze(1)
        # (D/2)
        div_term = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        # (N, D)
        pos_embd = torch.zeros((n, d))
        # (N, D)
        pos_embd[:, 0::2] = torch.sin(position * div_term) * scale_coef
        # (N, D)
        pos_embd[:, 1::2] = torch.cos(position * div_term) * scale_coef

        # (N, D)
        return pos_embd

    def forward(self, x):
        B = x.shape[0]
        target_inp_size = self.target_inp_h * self.target_inp_w
        target_out_size = self.target_out_h * self.target_out_w
        # x: (B, Ht * Wt, C)
        assert x.shape == (B, target_inp_size, self.channels)

        # (B, _Ht * _Wt, D)
        x = self.preprocessor(x)
        # x: (B, _Ht * _Wt, D)
        assert x.shape == (B, target_out_size, self.embed_dim)
        # (B, _Ht * _Wt, D)
        x = x + self.positional_embd.expand(B, -1, -1)

        for mam_block in self.mam_blocks:
            # (B, _Ht * _Wt, D)
            x = mam_block(x)

        # (B, _Ht * _Wt, D)
        return x


class ClassesHead(nn.Module):
    """
    Module prepended to Mixformer backbone to predict the proportion of each class on image.

    Input shape: (B, H * W, C)
    Output shape: (B, 5)
    """
    def __init__(self, config):
        super(ClassesHead, self).__init__()

        self.channels = config['channels']
        self.target_h = config['target_h']
        self.target_w = config['target_w']
        self.linear_size = config['linear_size']

        self.conv1 = nn.Conv2d(self.channels, self.channels // 2, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(self.channels // 2)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(self.channels // 2, self.linear_size)
        self.linear2 = nn.Linear(self.linear_size, 5)

    def forward(self, x):
        B = x.shape[0]
        # x: (B, H * W, C)
        assert x.shape == (B, self.target_h * self.target_w, self.channels)

        # (B, C, H, W)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.target_h, w=self.target_w).contiguous()
        # (B, C/2, H, W)
        x = F.selu(self.batchnorm1(self.conv1(x)))
        # (B, C/2)
        x = self.global_average_pool(x).squeeze(-1).squeeze(-1)
        # x: (B, C/2)
        assert x.shape == (B, self.channels // 2)
        # (B, LS)
        x = F.selu(self.linear1(x))
        # (B, 5)
        x = self.linear2(x)
        # x: (B, 5)
        assert x.shape == (B, 5)

        # (B, 5)
        return x


class MixFormer(nn.Module):
    """
    MixFormer model.

    Input shape: (B, Ht, Wt, 3)
    Output shape: (B, 5)
    """
    def __init__(self, config):
        super(MixFormer, self).__init__()
        
        self.target_inp_h = config['target_inp_h']
        self.target_inp_w = config['target_inp_w']
        self.target_out_h = config['target_out_h']
        self.target_out_w = config['target_out_w']
        self.out_embed_dim = config['out_embed_dim']
        self.num_stages = config['num_stages']

        self.stages = nn.ModuleList([Stage(config[f'stage_{i}']) for i in range(self.num_stages)])
        self.classes_head = ClassesHead(config['classes_head'])

    def forward(self, target):
        B = target.shape[0]
        target_inp_size = self.target_inp_h * self.target_inp_w
        target_out_size = self.target_out_h * self.target_out_w
        # target: (B, Ht, Wt, 3)
        assert target.shape == (B, self.target_inp_h, self.target_inp_w, 3)

        # (B, Ht * Wt, C)
        x = target.view(B, target_inp_size, 3)

        for stage in self.stages:
            # (B, _Ht * _Wt, D)
            x = stage(x)

        # x: (B, _Ht * _Wt, D)
        assert x.shape == (B, target_out_size, self.out_embed_dim)

        # (B, 5)
        output = self.classes_head(x)
        # output: (B, 5)
        assert output.shape == (B, 5)

        # (B, 5)
        return output
