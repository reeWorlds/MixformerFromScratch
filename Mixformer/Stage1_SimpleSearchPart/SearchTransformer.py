import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath


class StagePreprocessor(nn.Module):
    """
    Preprocess the search and target images to be used by Attention mechanism.

    Input shape: (B, Hs * Ws, C)
    Output shape: (B, N, D); N = _Hs * _Ws
    """
    def __init__(self, config, base_model=None):
        super(StagePreprocessor, self).__init__()

        self.channels = config['channels']
        self.embed_dim = config['embed_dim']
        self.search_inp_h = config['search_inp_h']
        self.search_inp_w = config['search_inp_w']
        self.search_out_h = config['search_out_h']
        self.search_out_w = config['search_out_w']
        self.patch_size = config['patch_size']
        self.patch_stride = config['patch_stride']
        self.patch_padding = config['patch_padding']

        if base_model is None:
            self.proj = nn.Conv2d(self.channels, self.embed_dim, self.patch_size, self.patch_stride, self.patch_padding)
            self.norm = nn.LayerNorm(self.embed_dim)
        else:
            self.proj = copy.deepcopy(base_model.proj)
            self.norm = copy.deepcopy(base_model.norm)

    def forward(self, x):
        B = x.shape[0]

        # x: (B, Hs * Ws, C)
        assert x.shape == (B, self.search_inp_h * self.search_inp_w, self.channels)

        # (B, C, Hs, Ws)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.search_inp_h, w=self.search_inp_w).contiguous()
        # (B, D, _Hs, _Ws)
        x = self.proj(x)
        # x: (B, D, _Hs, _Ws)
        assert x.shape == (B, self.embed_dim, self.search_out_h, self.search_out_w)
        # (B, _Hs * _Ws, D)
        x = rearrange(x, 'b d h w -> b (h w) d').contiguous()
        # (B, _Hs * _Ws, D)
        x = self.norm(x)

        # (B, N, D)
        return x


class DepthWiseQueryKeyValue(nn.Module):
    """
    Depth-wise CNN + query, key, value projection.

    Input shape: (B, N, C); N = Hs * Ws
    Output shape: (B, H, _Ns, C/H), (B, H, __Ns, C/H), (B, H, __Ns, C/H)
    """
    def __init__(self, config, base_model=None):
        super(DepthWiseQueryKeyValue, self).__init__()
        assert config['embed_dim'] % config['num_heads'] == 0
        
        self.embed_dim = config['embed_dim']
        self.search_inp_h = config['search_inp_h']
        self.search_inp_w = config['search_inp_w']
        self.search_q_h = config['search_q_h']
        self.search_q_w = config['search_q_w']
        self.search_kv_h = config['search_kv_h']
        self.search_kv_w = config['search_kv_w']
        self.kernel_size = config['kernel_size']
        self.padding_q = config['padding_q']
        self.stride_q = config['stride_q']
        self.padding_kv = config['padding_kv']
        self.stride_kv = config['stride_kv']
        self.num_heads = config['num_heads']
        self.head_dim = self.embed_dim // self.num_heads

        if base_model is None:
            self.norm1 = nn.LayerNorm(self.embed_dim)
            self.depthwise_q = self.build_conv_proj(self.embed_dim, self.kernel_size, self.padding_q, self.stride_q)
            self.depthwise_k = self.build_conv_proj(self.embed_dim, self.kernel_size, self.padding_kv, self.stride_kv)
            self.depthwise_v = self.build_conv_proj(self.embed_dim, self.kernel_size, self.padding_kv, self.stride_kv)
            self.proj_q = nn.Linear(self.embed_dim, self.embed_dim)
            self.proj_k = nn.Linear(self.embed_dim, self.embed_dim)
            self.proj_v = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            self.norm1 = copy.deepcopy(base_model.norm1)
            self.depthwise_q = copy.deepcopy(base_model.depthwise_q)
            self.depthwise_k = copy.deepcopy(base_model.depthwise_k)
            self.depthwise_v = copy.deepcopy(base_model.depthwise_v)
            self.proj_q = copy.deepcopy(base_model.proj_q)
            self.proj_k = copy.deepcopy(base_model.proj_k)
            self.proj_v = copy.deepcopy(base_model.proj_v)

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
        Ns = self.search_inp_h * self.search_inp_w

        # (B, N, C)
        x = self.norm1(x)

        # x: (B, Ns, C)
        assert x.shape == (B, Ns, self.embed_dim)

        # (B, C, Hs, Ws)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.search_inp_h, w=self.search_inp_w).contiguous()
        # (B, _Ns, C)
        search_q = self.depthwise_q(x)
        # (B, _Ns, C)
        assert search_q.shape == (B, self.search_q_h * self.search_q_w, self.embed_dim)
        # (B, __Ns, C)
        search_k = self.depthwise_k(x)
        # (B, __Ns, C)
        assert search_k.shape == (B, self.search_kv_h * self.search_kv_w, self.embed_dim)
        # (B, __Ns, C)
        search_v = self.depthwise_v(x)
        # (B, __Ns, C)
        assert search_v.shape == (B, self.search_kv_h * self.search_kv_w, self.embed_dim)
        # (B, _Ns, C)
        search_q = self.proj_q(search_q)
        # (B, H, _Ns, C/H)
        search_q = rearrange(search_q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim).contiguous()
        # (B, __Ns, C)
        search_k = self.proj_k(search_k)
        # (B, H, __Ns, C/H)
        search_k = rearrange(search_k, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim).contiguous()
        # (B, __Ns, C)
        search_v = self.proj_v(search_v)
        # (B, H, __Ns, C/H)
        search_v = rearrange(search_v, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_dim).contiguous()
        
        # (B, H, _Ns, D/H), (B, H, __Ns, D/H), (B, H, __Ns, D/H)
        return search_q, search_k, search_v


class MultiHeadAttention(nn.Module):
    """
    Asymetric Multi-Head Attention Described in the paper.
    
    Input shape: (B, N, D), (B, H, _Ns, D/H), (B, H, __Ns, D/H), (B, H, __Ns, D/H)
    Output shape: (B, N, D)
    """
    def __init__(self, config, base_model=None):
        super(MultiHeadAttention, self).__init__()
        assert config['embed_dim'] % config['num_heads'] == 0

        self.embed_dim = config['embed_dim']
        self.search_inp_h = config['search_inp_h']
        self.search_inp_w = config['search_inp_w']
        self.search_q_h = config['search_q_h']
        self.search_q_w = config['search_q_w']
        self.search_kv_h = config['search_kv_h']
        self.search_kv_w = config['search_kv_w']
        self.num_heads = config['num_heads']
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1 / (self.head_dim ** 0.5)
        self.ff_scale = config['ff_scale']

        if base_model is None:
            self.drop1 = DropPath(0.2)
            self.norm2 = nn.LayerNorm(self.embed_dim)
            self.drop2 = DropPath(0.2)
            self.ff_proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * self.ff_scale),
                nn.GELU(),
                nn.Linear(self.embed_dim * self.ff_scale, self.embed_dim)
            )
        else:
            self.drop1 = copy.deepcopy(base_model.drop1)
            self.norm2 = copy.deepcopy(base_model.norm2)
            self.drop2 = copy.deepcopy(base_model.drop2)
            self.ff_proj = copy.deepcopy(base_model.ff_proj)

    def forward(self, x, search_q, search_k, search_v):
        B = search_q.shape[0]

        # search_q: (B, H, _Ns, D/H)
        assert search_q.shape == (B, self.num_heads, self.search_q_h * self.search_q_w, self.head_dim)
        # search_k: (B, H, __Ns, D/H)
        assert search_k.shape == (B, self.num_heads, self.search_kv_h * self.search_kv_w, self.head_dim)
        # search_v: (B, H, __Ns, D/H)
        assert search_v.shape == (B, self.num_heads, self.search_kv_h * self.search_kv_w, self.head_dim)

        # (B, H, _Ns, __Ns)
        search_attn = torch.einsum('bhnd,bhmd->bhnm', [search_q, search_k]) * self.scale
        # (B, H, _Ns, __Ns)
        search_attn = F.softmax(search_attn, dim=-1)
        # (B, H, _Ns, D/H)
        search_attn = torch.einsum('bhnm,bhmd->bhnd', [search_attn, search_v])
        # (B, _Ns, D)
        search_attn = rearrange(search_attn, 'b h n d -> b n (h d)').contiguous()
        # (B, _Ns, D)
        assert search_attn.shape == (B, self.search_inp_h * self.search_inp_w, self.embed_dim)
        
        # (B, _Ns, D)
        x = x + self.drop1(search_attn)
        # (B, _Ns, D)
        x = x + self.drop1(self.ff_proj(self.norm2(x)))

        # (B, N, D)
        return x


class MixedAttentionModule(nn.Module):
    """
    Mixed Attention Module described in the paper.

    Input shape: (B, N, D)
    Output shape: (B, N, D)
    """
    def __init__(self, config, base_model=None):
        super(MixedAttentionModule, self).__init__()

        self.embed_dim = config['embed_dim']

        if base_model is None:
            self.depthwise_qkv = DepthWiseQueryKeyValue(config['depthwise_qkv'])
            self.attention = MultiHeadAttention(config['attention'])
        else:
            self.depthwise_qkv = DepthWiseQueryKeyValue(config['depthwise_qkv'], base_model.depthwise_qkv)
            self.attention = MultiHeadAttention(config['attention'], base_model.attention)

    def forward(self, x):
        # (B, H, _Ns, D/H), (B, H, __Ns, D/H), (B, H, __Ns, D/H)
        target_q, target_k, target_v = self.depthwise_qkv(x)
        # (B, N, D)
        x = self.attention(x, target_q, target_k, target_v)

        # (B, N, D)
        return x


class Stage(nn.Module):
    """
    One stage of the ConvolutionalVisioNsransformer (CVT) model.
    
    Input shape: (B, Hs * Ws, C)
    Output shape: (B, _Hs * _Ws, D)
    """
    def __init__(self, config, base_model=None):
        super(Stage, self).__init__()
        
        self.channels = config['channels']
        self.embed_dim = config['embed_dim']
        self.search_inp_h = config['search_inp_h']
        self.search_inp_w = config['search_inp_w']
        self.search_out_h = config['search_out_h']
        self.search_out_w = config['search_out_w']
        self.num_mam_blocks = config['num_mam_blocks']

        search_embd = self.get_pos_embd(self.search_out_h * self.search_out_w, self.embed_dim)
        self.register_buffer('positional_embd', search_embd)
        if base_model is None:
            self.preprocessor = StagePreprocessor(config['preprocessor'])
            self.mam_blocks = nn.ModuleList([MixedAttentionModule(config['mam']) for _ in range(self.num_mam_blocks)])
        else:
            self.preprocessor = StagePreprocessor(config['preprocessor'], base_model.preprocessor)
            self.mam_blocks = nn.ModuleList([MixedAttentionModule(config['mam'], base_model.mam_blocks[i])
                                             for i in range(self.num_mam_blocks)])

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
        search_inp_size = self.search_inp_h * self.search_inp_w
        search_out_size = self.search_out_h * self.search_out_w
        # x: (B, Hs * Ws, C)
        assert x.shape == (B, search_inp_size, self.channels)

        # (B, _Hs * _Ws, D)
        x = self.preprocessor(x)
        # x: (B, _Hs * _Ws, D)
        assert x.shape == (B, search_out_size, self.embed_dim)
        # (B, _Hs * _Ws, D)
        x = x + self.positional_embd.expand(B, -1, -1)

        for mam_block in self.mam_blocks:
            # (B, _Hs * _Ws, D)
            x = mam_block(x)

        # (B, _Hs * _Ws, D)
        return x


class MaskHead(nn.Module):
    """
    Module prepended to Mixformer backbone to predict class of each pixel.

    Input shape: (B, H * W, C), (B)
    Output shape: (B, _H, _W)
    """
    def __init__(self, config, base_model=None):
        super(MaskHead, self).__init__()
        assert base_model is None

        self.channels = config['channels']
        self.search_inp_h = config['search_inp_h']
        self.search_inp_w = config['search_inp_w']
        self.search_out_h = config['search_out_h']
        self.search_out_w = config['search_out_w']

        self.deconv1 = self.make_deconvolution_block(self.channels)
        self.deconv2 = self.make_deconvolution_block(self.channels // 2)
        self.conv3 = nn.Conv2d(self.channels // 4, self.channels // 4, 3, 1, 1)
        self.linear = nn.Linear(self.channels // 4, 1)

    def make_deconvolution_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(channels // 2, channels // 2, 3, 1, 1),
            nn.BatchNorm2d(channels // 2),
            nn.SELU(inplace=True)
        )

    def forward(self, x):
        B = x.shape[0]
        # x: (B, H * W, C)
        assert x.shape == (B, self.search_inp_h * self.search_inp_w, self.channels)

        # (B, C, H, W)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.search_inp_h, w=self.search_inp_w).contiguous()
        # (B, C/2, 2*H, 2*W)
        x = self.deconv1(x)
        # (B, C/4, 4*H, 4*W)
        x = self.deconv2(x)
        # (B, C/4, 4*H, 4*W)
        x = F.selu(self.conv3(x))
        # (B, 4*h, 4*W, C/4)
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        # (B, 4*H, 4*W, 1)
        x = self.linear(x)
        # (B, _H, _W, 1)
        assert x.shape == (B, self.search_out_h, self.search_out_w, 1)

        # (B, _H, _W)
        x = x.squeeze(3)

        # (B, _H, _W)
        return x


class MixFormer(nn.Module):
    """
    MixFormer model.

    Input shape: (B, Hs, Ws, 3)
    Output shape: (B, Hs, Ws)
    """
    def __init__(self, config, base_model=None):
        super(MixFormer, self).__init__()
        
        self.proj_channels = config['proj_channels']
        self.search_inp_h = config['search_inp_h']
        self.search_inp_w = config['search_inp_w']
        self.search_out_h = config['search_out_h']
        self.search_out_w = config['search_out_w']
        self.out_embed_dim = config['out_embed_dim']
        self.num_stages = config['num_stages']
        
        self.proj_cnn = nn.Conv2d(3, self.proj_channels, 3, 1, 1)
        self.class_type_embeding = nn.Embedding(5, self.proj_channels)
        if base_model is None:
            self.stages = nn.ModuleList([Stage(config[f'stage_{i}']) for i in range(self.num_stages)])
        else:
            self.stages = nn.ModuleList([Stage(config[f'stage_{i}'], base_model.stages[i])
                                         for i in range(self.num_stages)])
        self.mask_head = MaskHead(config['mask_head'])

    def forward(self, search, class_index):
        B = search.shape[0]
        search_inp_size = self.search_inp_h * self.search_inp_w
        search_out_size = self.search_out_h * self.search_out_w
        # search: (B, Hs, Ws, 3)
        assert search.shape == (B, self.search_inp_h, self.search_inp_w, 3)

        # (B, 3, Hs, Ws)
        x = rearrange(search, 'b h w c -> b c h w').contiguous()
        # B, _C, Hs, Ws)
        search = self.proj_cnn(search)
        # (B, Hs * Ws, _C)
        x = rearrange(search, 'b c h w -> b (h w) c').contiguous()

        # (B, Hs * Ws, _C)
        cls = self.class_type_embeding(class_index).unsqueeze(1).expand(-1, search_inp_size, -1)
        # (B, Hs * Ws, _C)
        x = x + cls

        for stage in self.stages:
            # (B, _Hs * _Ws, D)
            x = stage(x)

        # x: (B, _Hs * _Ws, D)
        assert x.shape == (B, search_out_size, self.out_embed_dim)

        # (B, _Hs, _Ws)
        output = self.mask_head(x)
        # output: (B, Hs, Ws)
        assert output.shape == (B, self.search_inp_h, self.search_inp_w)

        # (B, Hs, Ws)
        return output

    def set_base_requires_grad(self, requires_grad=False):
        for stage in self.stages:
            self._set_requires_grad(stage.preprocessor.proj, requires_grad)
            self._set_requires_grad(stage.preprocessor.norm, requires_grad)
            for mam_block in stage.mam_blocks:
                self._set_requires_grad(mam_block.depthwise_qkv.norm1, requires_grad)
                self._set_requires_grad(mam_block.depthwise_qkv.depthwise_q, requires_grad)
                self._set_requires_grad(mam_block.depthwise_qkv.depthwise_k, requires_grad)
                self._set_requires_grad(mam_block.depthwise_qkv.depthwise_v, requires_grad)
                self._set_requires_grad(mam_block.depthwise_qkv.proj_q, requires_grad)
                self._set_requires_grad(mam_block.depthwise_qkv.proj_k, requires_grad)
                self._set_requires_grad(mam_block.depthwise_qkv.proj_v, requires_grad)
                self._set_requires_grad(mam_block.attention.drop1, requires_grad)
                self._set_requires_grad(mam_block.attention.norm2, requires_grad)
                self._set_requires_grad(mam_block.attention.drop2, requires_grad)
                self._set_requires_grad(mam_block.attention.ff_proj, requires_grad)

    def _set_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            param.requires_grad = requires_grad
