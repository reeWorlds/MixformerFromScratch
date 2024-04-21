import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath


class StagePreprocessor(nn.Module):
    """
    Preprocess the image to be used by attention mechanism.

    Input shape: (B, C * H, W)
    Output shape: (B, N, _C); N = [1 +] _H * _W
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
        self.search_target_embd = nn.Embedding(2, self.in_c)
        self.search_target_embd.weight.data *= 0.2
        self.proj = nn.Conv2d(self.in_c, self.out_c, self.patch_size, self.patch_stride, self.patch_padding)
        if self.use_cls:
            self.register_parameter('cls_token', nn.Parameter(torch.randn(1, 1, self.out_c)))

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

    def forward(self, x):
        B = x.shape[0]
        HW = math.isqrt(x.shape[1])

        # x: (B, H * W, C)
        assert x.shape == (B, HW * HW, self.in_c)

        if HW == self.search_hw:
            # (B, H * W, C)
            x = x + self.search_pos_embd.view(1, -1, self.in_c)
            # (1)
            ind = torch.zeros(1, dtype=torch.int32).to(x.device)
            # (B, H * W, C)
            x = x + self.search_target_embd(ind).view(1, 1, self.in_c)
        else:
            # (B, H * W, C)
            x = x + self.target_pos_embd.view(1, -1, self.in_c)
            # (1)
            ind = torch.ones(1, dtype=torch.int32).to(x.device)
            # (B, H * W, C)
            x = x + self.search_target_embd(ind).view(1, 1, self.in_c)

        # (B, C, H, W)
        x = rearrange(x, 'b (h w) c -> b c h w', h=HW, w=HW).contiguous()
        # (B, _C, _H, _W)
        x = self.proj(x)
        # (B, _H * _W, _C)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

        if self.use_cls:
            # (B, 1 + _H * _W, _C)
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        # (B, N, _C)
        return x


class DepthWiseQueryKeyValue(nn.Module):
    """
    Depth-wise CNN + query, key, value projection.

    Input shape: (B, N, C); N = [1 +] H * W
    Output shape: (B, H, [1 +] _N, C/H), (B, H, [1 +] __N, C/H), (B, H, [1 +] __N, C/H)
    """
    def __init__(self, config):
        super(DepthWiseQueryKeyValue, self).__init__()
        assert config['embd_d'] % config['num_heads'] == 0

        self.embd_d = config['embd_d']
        self.kernel_size = config['kernel_size']
        self.padding_q = config['padding_q']
        self.stride_q = config['stride_q']
        self.padding_kv = config['padding_kv']
        self.stride_kv = config['stride_kv']
        self.num_heads = config['num_heads']
        self.head_dim = self.embd_d // self.num_heads
        self.use_cls = config['use_cls']

        self.norm1 = nn.LayerNorm(self.embd_d)
        self.depthwise_q = self.build_conv_proj(self.embd_d, self.kernel_size, self.padding_q, self.stride_q)
        self.depthwise_k = self.build_conv_proj(self.embd_d, self.kernel_size, self.padding_kv, self.stride_kv)
        self.depthwise_v = self.build_conv_proj(self.embd_d, self.kernel_size, self.padding_kv, self.stride_kv)
        self.proj_q = nn.Linear(self.embd_d, self.embd_d)
        self.proj_k = nn.Linear(self.embd_d, self.embd_d)
        self.proj_v = nn.Linear(self.embd_d, self.embd_d)

    def build_conv_proj(self, channels, kernel_size, padding, stride):
        proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels),
            nn.BatchNorm2d(channels),
            Rearrange('b c h w -> b (h w) c')
        )
        return proj

    def forward(self, x):
        cls_sz = 1 if self.use_cls else 0
        HW = math.isqrt(x.shape[1] - cls_sz)

        # (B, N, C)
        x = self.norm1(x)
        if self.use_cls:
            # (B, 1, C), (B, N, C)
            cls, image = torch.split(x, [1, HW * HW], dim=1)
        else:
            # (B, N, C)
            image = x

        # (B, C, H, W)
        image = rearrange(image, 'b (h w) c -> b c h w', h=HW, w=HW).contiguous()
        # (B, _N, C)
        image_q = self.depthwise_q(image)
        # (B, __N, C)
        image_k = self.depthwise_k(image)
        # (B, __N, C)
        image_v = self.depthwise_v(image)
        if self.use_cls:
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

        # (B, H, [1 +] _N, D/H), (B, H, [1 +] __N, D/H), (B, H, [1 +] __N, D/H)
        return image_q, image_k, image_v


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.

    Input shape: (B, N, D), (B, H, [1 +] _N, D/H), (B, H, [1 +] __N, D/H), (B, H, [1 +] __N, D/H)
    Output shape: (B, N, D)
    """
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
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

    def forward(self, x, image_q, image_k, image_v):
        # (B, H, [1 +] _N, [1 +] __N)
        attn = torch.einsum('bhnd,bhmd->bhnm', [image_q, image_k]) * self.scale
        # (B, H, [1 +] _N, [1 +] __N)
        attn = F.softmax(attn, dim=-1)
        # (B, H, [1 +] _N, D/H)
        attn = torch.einsum('bhnm,bhmd->bhnd', [attn, image_v])
        # (B, [1 +] _N, D)
        attn = rearrange(attn, 'b h n d -> b n (h d)').contiguous()

        # (B, [1 +] _N, D)
        x = x + self.drop1(attn)
        # (B, [1 +] _N, D)
        x = x + self.drop1(self.ff_proj(self.norm2(x)))

        # (B, N, D)
        return attn


class MixedAttentionModule(nn.Module):
    """
    Just kind of Transformer block during pretraining.

    Input shape: (B, N, D)
    Output shape: (B, N, D)
    """
    def __init__(self, config):
        super(MixedAttentionModule, self).__init__()

        self.embd_d = config['embd_d']

        self.depthwise_qkv = DepthWiseQueryKeyValue(config['depthwise_qkv'])
        self.attention = MultiHeadAttention(config['attention'])
        self.final_proj = nn.Linear(self.embd_d, self.embd_d)

    def forward(self, x):
        # (B, H, [1 +] _N, D/H), (B, H, [1 +] __N, D/H), (B, H, [1 +] __N, D/H)
        image_q, image_k, imageh_v = self.depthwise_qkv(x)
        # (B, N, D)
        x = self.attention(x, image_q, image_k, imageh_v)
        # (B, N, D)
        x = x + self.final_proj(x)

        # (B, N, D)
        return x


class Stage(nn.Module):
    """
    One stage of the ConvolutionalVisionTransformer (CVT) model.

    Input shape: (B, H * W, C)
    Output shape: (B, [1 +] _H * _W, D)
    """
    def __init__(self, config):
        super(Stage, self).__init__()

        self.num_mams = config['num_mams']

        self.preprocessor = StagePreprocessor(config['preprocessor'])
        self.mam_blocks = nn.ModuleList([MixedAttentionModule(config['mam']) for _ in range(self.num_mams)])

    def forward(self, x):
        # (B, [1 +] _H * _W, D)
        x = self.preprocessor(x)

        for mam_block in self.mam_blocks:
            # (B, [1 +] _H * _W, D)
            x = mam_block(x)

        # (B, [1 +] _H * _W, D)
        return x


class MaskHead(nn.Module):
    """
    Module prepended to Transformer backbone to predict class of each pixel.

    Input shape: (B, _H, _W, C)
    Output shape: (B, H, W, 5)
    """
    def __init__(self, config):
        super(MaskHead, self).__init__()

        self.channels = config['channels']

        self.conv1_1 = nn.Conv2d(self.channels, self.channels // 2, 3, padding=1)
        self.conv1_2 = nn.Conv2d(self.channels // 2, self.channels * 2, 1, padding=0)
        self.conv1_3 = nn.PixelShuffle(2)

        self.conv2_1 = nn.Conv2d(self.channels // 2, self.channels // 4, 3, padding=1)
        self.conv2_2 = nn.Conv2d(self.channels // 4, self.channels, 1, padding=0)
        self.conv2_3 = nn.PixelShuffle(2)

        self.conv3_1 = nn.Conv2d(self.channels // 4, self.channels // 4, 3, padding=1)
        self.conv3_2 = nn.Conv2d(self.channels // 4, 5, 1, padding=0)

    def forward(self, x):
        # (B, C, _H, _W)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        # (B, C, 2 * _H, 2 * _W)
        x = self.conv1_3(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))
        # (B, C, 4 * _H, 4 * _W)
        x = self.conv2_3(F.relu(self.conv2_2(F.relu(self.conv2_1(x)))))
        # (B, 5, 4 * _H, 4 * _W)
        x = self.conv3_2(F.relu(self.conv3_1(x)))
        # (B, 4 * _H, 4 * _W, 5)
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        # (B, H, W, 5)
        x = F.softmax(x, dim=-1)

        # (B, H, W, 5)
        return x


class ClassHead(nn.Module):
    """
    Module appended to Transformer backbone to predict the class of the image.

    Input shape: (B, D)
    Output shape: (B, 20)
    """
    def __init__(self, config):
        super(ClassHead, self).__init__()

        self.embd_d = config['embd_d']
        self.inner_dim = config['inner_dim']

        self.linear1 = nn.Linear(self.embd_d, self.inner_dim)
        self.linear2 = nn.Linear(self.inner_dim, 20)

    def forward(self, x):
        # (B, D)
        x = F.relu(self.linear1(x))
        # (B, 20)
        x = F.softmax(self.linear2(x), dim=-1)

        # (B, 20)
        return x


class Transformer(nn.Module):
    """
    Kind of Transmormer model.

    Input shape: (B, H, W, 3)
    Output shape: (B, H, W, 5), (B, 20)
    """
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.proj_channels = config['proj_channels']
        self.num_stages = config['num_stages']

        self.init_projection = nn.Conv2d(3, self.proj_channels, 3, padding=1)
        self.stages = nn.ModuleList([Stage(config[f'stage_{i}']) for i in range(self.num_stages)])
        self.mask_head = MaskHead(config['mask_head'])
        self.class_head = ClassHead(config['class_head'])

    def forward(self, x):
        # (B, 3, H, W)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        # (B, C, H, W)
        x = F.relu(self.init_projection(x))
        # (B, H * W, C)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

        for stage in self.stages:
            # (B, [1 +] _H * _W, _D)
            x = stage(x)

        HW = math.isqrt(x.shape[1] - 1)

        # (B, 1, D), (B, _H * _W, D)
        cls, image = torch.split(x, [1, HW * HW], dim=1)
        # (B, D)
        cls = cls.squeeze(1)
        # (B, _H, _W, D)
        image = rearrange(image, 'b (h w) c -> b h w c', h=HW, w=HW).contiguous()

        # (B, H, W, 5)
        res_mask = self.mask_head(image)
        # (B, 20)
        res_class = self.class_head(cls)

        # (B, H, W, 5), (B, 20)
        return res_mask, res_class
