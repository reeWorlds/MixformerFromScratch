import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class StagePreprocessor(nn.Module):
    """
    Preprocess search and target images to be used by attention mechanism.

    Input shape: (B, Hs * Ws + Ht * Wt, C)
    Output shape: (B, N, _C); N = [1 +] _Hs * _Ws + _Ht * _Wt
    """
    def __init__(self, config):
        super(StagePreprocessor, self).__init__()

        self.search_hw = config['search_hw']
        self.target_hw = config['target_hw']
        self.in_c = config['in_c']
        self.out_c = config['out_c']
        self.patch_size = config['patch_size']
        self.patch_stride = config['patch_stride']
        self.patch_padding = config['patch_padding']
        self.use_cls = config['use_cls']

        search_pos_embd = self.get_pos_embd(self.search_hw * self.search_hw, self.in_c, 10000.0)
        self.register_buffer('search_pos_embd', search_pos_embd)
        target_pos_embd = self.get_pos_embd(self.target_hw * self.target_hw, self.in_c, 2000.0)
        self.register_buffer('target_pos_embd', target_pos_embd)
        self.register_parameter('search_embd', nn.Parameter(torch.randn(1, 1, self.in_c) * 0.2))
        self.register_parameter('target_embd', nn.Parameter(torch.randn(1, 1, self.in_c) * 0.2))
        self.proj = nn.Conv2d(self.in_c, self.out_c, self.patch_size, self.patch_stride, self.patch_padding)
        if self.use_cls:
            self.register_parameter('cls_token', nn.Parameter(torch.randn(1, 1, self.out_c)))
        self.norm = nn.LayerNorm(self.out_c)

    def get_pos_embd(self, n, d, freq):
        scale_coef = 0.3

        # (N, 1)
        position = torch.arange(n).unsqueeze(1)
        # (D/2)
        div_term = torch.exp(torch.arange(0, d, 2).float() * -(math.log(freq) / d))
        # (N, D)
        pos_embd = torch.zeros((n, d))
        # (N, D)
        pos_embd[:, 0::2] = torch.sin(position * div_term) * scale_coef
        # (N, D)
        pos_embd[:, 1::2] = torch.cos(position * div_term) * scale_coef

        # (N, D)
        return pos_embd

    def forward(self, search_target):
        B = search_target.shape[0]
        SHW2, THW2 = self.search_hw * self.search_hw, self.target_hw * self.target_hw

        # search_target: (B, Hs * Ws + Ht * Wt, C)
        assert search_target.shape == (B, SHW2 + THW2, self.in_c)
        # (B, Hs * Ws, C), (B, Ht * Wt, C)
        search, target = torch.split(search_target, [SHW2, THW2], dim=1)

        # (B, Hs * Ws, C)
        search = search + self.search_pos_embd.view(1, SHW2, self.in_c) + self.search_embd
        # (B, Ht * Wt, C)
        target = target + self.target_pos_embd.view(1, THW2, self.in_c) + self.target_embd

        # (B, C, Hs, Ws)
        search = rearrange(search, 'b (h w) c -> b c h w', h=self.search_hw, w=self.search_hw).contiguous()
        # (B, _C, _Hs, _Ws)
        search = self.proj(search)
        # (B, _Hs * _Ws, _C)
        search = rearrange(search, 'b c h w -> b (h w) c').contiguous()

        # (B, C, Ht, Wt)
        target = rearrange(target, 'b (h w) c -> b c h w', h=self.target_hw, w=self.target_hw).contiguous()
        # (B, _C, _Ht, _Wt)
        target = self.proj(target)
        # (B, _Ht * _Wt, _C)
        target = rearrange(target, 'b c h w -> b (h w) c').contiguous()

        if self.use_cls:
            # (B, 1 + _Hs * _Ws + _Ht * _Wt, _C)
            x = torch.cat([self.cls_token.expand(B, -1, -1), search, target], dim=1)
        else:
            # (B, _Hs * _Ws + _Ht * _Wt, _C)
            x = torch.cat([search, target], dim=1)

        # (B, N, _C)
        x = self.norm(x)

        # (B, N, _C)
        return x


class DepthWiseQueryKeyValue(nn.Module):
    """
    Depth-wise CNN + query, key, value projection.

    Input shape: (B, N, C); N = [1 +] Hs * Ws + Ht * Wt
    Output shape: (B, H, [1 +] _Ns, C/H), (B, H, [1 +] __Ns, C/H), (B, H, [1 +] __Ns, C/H)
                (B, H, _Nt, C/H), (B, H, __Nt, C/H), (B, H, __Nt, C/H)
    """
    def __init__(self, config):
        super(DepthWiseQueryKeyValue, self).__init__()
        assert config['embd_d'] % config['num_heads'] == 0

        self.embd_d = config['embd_d']
        self.search_hw = config['search_hw']
        self.target_hw = config['target_hw']
        self.kernel_size = config['kernel_size']
        self.padding_q = config['padding_q']
        self.stride_q = config['stride_q']
        self.padding_kv = config['padding_kv']
        self.stride_kv = config['stride_kv']
        self.num_heads = config['num_heads']
        self.head_dim = self.embd_d // self.num_heads
        self.use_cls = config['use_cls']

        #self.norm1 = nn.LayerNorm(self.embd_d)
        self.norm1 = Identity()
        self.depthwise_q = self.conv_proj(self.embd_d, self.kernel_size, self.padding_q, self.stride_q)
        self.depthwise_k = self.conv_proj(self.embd_d, self.kernel_size, self.padding_kv, self.stride_kv)
        self.depthwise_v = self.conv_proj(self.embd_d, self.kernel_size, self.padding_kv, self.stride_kv)
        self.proj_q = nn.Linear(self.embd_d, self.embd_d)
        self.proj_k = nn.Linear(self.embd_d, self.embd_d)
        self.proj_v = nn.Linear(self.embd_d, self.embd_d)

    def conv_proj(self, channels, kernel_size, padding, stride):
        proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels),
            nn.BatchNorm2d(channels),
            Rearrange('b c h w -> b (h w) c')
        )
        return proj

    def forward(self, x):
        B = x.shape[0]
        cls_sz = 1 if self.use_cls else 0
        SHW2, THW2 = self.search_hw * self.search_hw, self.target_hw * self.target_hw

        # x: (B, [1 +] Hs * Ws + Ht * Wt, C)
        assert x.shape == (B, cls_sz + SHW2 + THW2, self.embd_d)

        # (B, N, C)
        x = self.norm1(x)

        if self.use_cls:
            # (B, 1, C), (B, Hs * Ws, C), (B, Ht * Wt, C)
            cls, search, target = torch.split(x, [1, SHW2, THW2], dim=1)
        else:
            # (B, Hs * Ws, C), (B, Ht * Wt, C)
            search, target = torch.split(x, [SHW2, THW2], dim=1)

        if self.use_cls:
            # (B, H, [1 +] _Ns, C/H), (B, H, [1 +] __Ns, C/H), (B, H, [1 +] __Ns, C/H)
            search_q, search_k, search_v = self.get_qkv(search, cls)
        else:
            # (B, H, _Ns, C/H), (B, H, __Ns, C/H), (B, H, __Ns, C/H)
            search_q, search_k, search_v = self.get_qkv(search)

        # (B, H, _Nt, C/H), (B, H, __Nt, C/H), (B, H, __Nt, C/H)
        target_q, target_k, target_v = self.get_qkv(target)

        # (B, H, [1 +] _Ns, D/H), (B, H, [1 +] __Ns, D/H), (B, H, [1 +] __Ns, D/H)
        # (B, H, _Nt, D/H), (B, H, __Nt, D/H), (B, H, __Nt, D/H)
        return search_q, search_k, search_v, target_q, target_k, target_v

    def get_qkv(self, image, cls=None):
        HW = math.isqrt(image.shape[1])

        # (B, C, H, W)
        image = rearrange(image, 'b (h w) c -> b c h w', h=HW, w=HW).contiguous()
        # (B, _N, C)
        image_q = self.depthwise_q(image)
        # (B, __N, C)
        image_k = self.depthwise_k(image)
        # (B, __N, C)
        image_v = self.depthwise_v(image)

        if cls is not None:
            # (B, 1 + _N, C)
            image_q = torch.cat([cls, image_q], dim=1)
            # (B, 1 + __N, C)
            image_k = torch.cat([cls, image_k], dim=1)
            # (B, 1 + __N, C)
            image_v = torch.cat([cls, image_v], dim=1)

        # (B, [1 +] _N, C)
        image_q = self.proj_q(image_q)
        # (B, H, [1 +] _N, C/H)
        image_q = rearrange(image_q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim).contiguous()
        # (B, [1 +] __N, C)
        image_k = self.proj_k(image_k)
        # (B, H, [1 +] __N, C/H)
        image_k = rearrange(image_k, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim).contiguous()
        # (B, [1 +] __N, C)
        image_v = self.proj_v(image_v)
        # (B, H, [1 +] __N, C/H)
        image_v = rearrange(image_v, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim).contiguous()

        return image_q, image_k, image_v


class AsymetricMultiHeadAttention(nn.Module):
    """
    Asymetrical Multi-Head Attention.

    Input shape: (B, N, D), (B, H, [1 +] _Ns, D/H), (B, H, [1 +] __Ns, D/H), (B, H, [1 +] __Ns, D/H)
                (B, H, _Nt, D/H), (B, H, __Nt, D/H), (B, H, __Nt, D/H)
    Output shape: (B, N, D)
    """
    def __init__(self, config):
        super(AsymetricMultiHeadAttention, self).__init__()
        assert config['embd_d'] % config['num_heads'] == 0

        self.embd_d = config['embd_d']
        self.num_heads = config['num_heads']
        self.head_dim = self.embd_d // self.num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.ff_scale = config['ff_scale']

        self.drop1 = DropPath(0.2)
        self.norm2 = nn.LayerNorm(self.embd_d)
        self.drop2 = DropPath(0.2)
        self.ff_proj = nn.Sequential(
            nn.Linear(self.embd_d, self.embd_d * self.ff_scale),
            nn.GELU(),
            nn.Linear(self.embd_d * self.ff_scale, self.embd_d)
        )

    def forward(self, x, search_q, search_k, search_v, target_q, target_k, target_v):
        # (B, H, [1 +] _Ns + _Nt, D/H), (B, H, [1 +] __Ns + __Nt, D/H), (B, H, [1 +] __Ns + __Nt, D/H)
        mixed_k = torch.cat([search_k, target_k], dim=2)
        # (B, H, [1 +] _Ns + _Nt, D/H), (B, H, [1 +] __Ns + __Nt, D/H), (B, H, [1 +] __Ns + __Nt, D/H)
        mixed_v = torch.cat([search_v, target_v], dim=2)

        # (B, [1 +] _Ns + _Nt, D)]
        search_attn = self.calc_attn(search_q, mixed_k, mixed_v)
        # (B, [1 +] __Ns + __Nt, D)]
        target_attn = self.calc_attn(target_q, target_k, target_v)

        attn = torch.cat([search_attn, target_attn], dim=1)

        # (B, [1 +] _N, D)
        x = x + self.drop1(attn)
        # (B, [1 +] _N, D)
        x = x + self.drop1(self.ff_proj(self.norm2(x)))

        # (B, N, D)
        return x
    
    def calc_attn(self, q, k, v):
        # (B, H, N, N)
        attn = torch.einsum('bhnd,bhmd->bhnm', [q, k]) * self.scale
        # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        # (B, H, N, D/H)
        attn = torch.einsum('bhnm,bhmd->bhnd', [attn, v])
        # (B, N, D)
        attn = rearrange(attn, 'b h n d -> b n (h d)').contiguous()
        
        # (B, N, D)
        return attn


class MixedAttentionModule(nn.Module):
    """
    MAM from the paper.

    Input shape: (B, N, D)
    Output shape: (B, N, D)
    """
    def __init__(self, config):
        super(MixedAttentionModule, self).__init__()

        self.embd_d = config['embd_d']

        self.depthwise_qkv = DepthWiseQueryKeyValue(config['depthwise_qkv'])
        self.attention = AsymetricMultiHeadAttention(config['attention'])

    def forward(self, x):
        # (B, H, [1 +] _Ns, D/H), (B, H, [1 +] __Ns, D/H), (B, H, [1 +] __Ns, D/H)
        # (B, H, _Nt, D/H), (B, H, __Nt, D/H), (B, H, __Nt, D/H)
        attn = self.depthwise_qkv(x)
        # (B, N, D)
        x = self.attention(x, *attn)

        # (B, N, D)
        return x


class Stage(nn.Module):
    """
    One stage of the ConvolutionalVisionTransformer (CVT) model.

    Input shape: (B, Hs * Ws + Ht * Wt, C)
    Output shape: (B, [1 +] _Hs * _Ws + _Ht * _Wt, D)
    """
    def __init__(self, config):
        super(Stage, self).__init__()

        self.num_mams = config['num_mams']

        self.preprocessor = StagePreprocessor(config['preprocessor'])
        self.mam_blocks = nn.ModuleList([MixedAttentionModule(config['mam']) for _ in range(self.num_mams)])

    def forward(self, x):
        # (B, [1 +] _Hs * _Ws + _Ht * _Wt, C)
        x = self.preprocessor(x)

        for mam_block in self.mam_blocks:
            # (B, [1 +] _Hs * _Ws + _Ht * _Wt, D)
            x = mam_block(x)

        # (B, [1 +] _Hs * _Ws + _Ht * _Wt, D)
        return x


class PositionHead(nn.Module):
    """
    Module prepended to MixFormer backbone to predict position on search image.

    Input shape: (B, _Hs, _Ws, C)
    Output shape: (B, Hs, Ws)
    """
    def __init__(self, config):
        super(PositionHead, self).__init__()

        self.channels = config['channels']

        self.conv1 = nn.Conv2d(self.channels, self.channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.conv2 = nn.Conv2d(self.channels, self.channels, 1)
        self.bn2 = nn.BatchNorm2d(self.channels)
        self.conv3 = nn.Conv2d(self.channels, 4, 1)
        self.linear1 = nn.Linear(16 * 16 * 4, 32)
        self.linear2 = nn.Linear(32, 2)

    def forward(self, x):
        # (B, C, H, W)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        # (B, C, H, W)
        x = F.selu(self.bn1(self.conv1(x)))
        # (B, C, H, W)
        x = F.selu(self.bn2(self.conv2(x)))
        # (B, 4, H, W)
        x = F.selu(self.conv3(x))
        # (B, 4 * H * W)
        x = rearrange(x, 'b c h w -> b (h w c)').contiguous()
        # (B, 32)
        x = F.selu(self.linear1(x))
        # (B, 2)
        x = self.linear2(x)

        # (B, 2)
        return x


class ScaleHead(nn.Module):
    """
    Module prepended to MixFormer backbone to predict scale of target image baed on cls token.

    Input shape: (B, C)
    Output shape: (B, 1)
    """
    def __init__(self, config):
        super(ScaleHead, self).__init__()

        self.channels = config['channels']

        self.linear1 = nn.Linear(self.channels, 32)
        self.linear2 = nn.Linear(32, 1)

    def forward(self, x):
        # (B, 32)
        x = F.selu(self.linear1(x))
        # (B, 1)
        x = self.linear2(x)

        # (B, 1)
        return x


class MixFormer(nn.Module):
    """
    MixFormer model.

    Input shape: (B, Hs, Ws, 3), (B, Hs, Wt, 3)
    Output shape: (B, 3)
    """
    def __init__(self, config):
        super(MixFormer, self).__init__()

        self.proj_channels = config['proj_channels']
        self.search_out_hw = config['search_out_hw']
        self.target_out_hw = config['target_out_hw']
        self.num_stages = config['num_stages']

        self.init_projection = nn.Conv2d(3, self.proj_channels, 3, padding=1)
        self.stages = nn.ModuleList([Stage(config[f'stage_{i}']) for i in range(self.num_stages)])
        self.position_head = PositionHead(config['position_head'])
        self.scale_head = ScaleHead(config['scale_head'])

    def forward(self, search, target):
        # (B, 3, Hs, Ws)
        search = rearrange(search, 'b h w c -> b c h w').contiguous()
        # (B, C, Hs, Ws)
        search = F.selu(self.init_projection(search))
        # (B, Hs * Ws, C)
        search = rearrange(search, 'b c h w -> b (h w) c').contiguous()

        # (B, 3, Ht, Wt)
        target = rearrange(target, 'b h w c -> b c h w').contiguous()
        # (B, C, Ht, Wt)
        target = F.selu(self.init_projection(target))
        # (B, Ht * Wt, C)
        target = rearrange(target, 'b c h w -> b (h w) c').contiguous()

        # (B, Hs * Ws + Ht * Wt, C)
        x = torch.cat([search, target], dim=1)

        for stage in self.stages:
            # (B, [1 +] _Hs * _Ws + _Ht * _Wt, D)
            x = stage(x)

        SHW2 = self.search_out_hw * self.search_out_hw
        THW2 = self.target_out_hw * self.target_out_hw

        # (B, 1, D), (B, _Hs * _Ws, D), (B, _Ht * _Wt, D)
        cls, search, target = torch.split(x, [1, SHW2, THW2], dim=1)
        # (B, D)
        cls = cls.squeeze(1)
        # (B, _Hs, _Ws, D)
        search = rearrange(search, 'b (h w) c -> b h w c', h=self.search_out_hw,
                           w=self.search_out_hw).contiguous()
        # (B, _Ht, _Wt, D)
        target = rearrange(target, 'b (h w) c -> b h w c', h=self.target_out_hw,
                           w=self.target_out_hw).contiguous()
        
        # (B, 2)
        position = self.position_head(search)
        # (B, 1)
        scale = self.scale_head(cls)

        # (B, 3)
        result = torch.cat([position, scale], dim=1)

        # (B, 3)
        return result
