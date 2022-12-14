import logging
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from slowfast.models.common import DropPath


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3, act=False, normtype=None):
        super(ConvPosEnc, self).__init__()
        assert normtype is None
        padding = k // 2
        self.proj = nn.Conv2d(dim, dim, (k, k), (1, 1), (padding, padding), groups=dim)
        self.proj_th = nn.Conv2d(dim, dim, (k, k), (1, 1), (padding, padding), groups=dim)
        self.proj_tw = nn.Conv2d(dim, dim, (k, k), (1, 1), (padding, padding), groups=dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        D, H, W = size
        assert N == D * H * W

        feat = rearrange(x, 'b (d h w) c -> (b d) c h w', d=D, h=H)
        feat = self.proj(feat)
        feat = rearrange(feat, '(b d) c h w -> b (d h w) c', d=D)

        feat_tw = rearrange(x, 'b (d h w) c -> (b h) c d w', d=D, h=H)
        feat_tw = self.proj_tw(feat_tw)
        feat_tw = rearrange(feat_tw, '(b h) c d w -> b (d h w) c', h=H)

        feat_th = rearrange(x, 'b (d h w) c -> (b w) c d h', d=D, h=H)
        feat_th = self.proj_th(feat_th)
        feat_th = rearrange(feat_th, '(b w) c d h -> b (d h w) c', w=W)

        feat = feat + feat_th + feat_tw
        x = x + self.activation(feat)
        return x


class ConvEmbed(nn.Module):
    """ Video to Patch Embedding
    """

    def __init__(
            self,
            patch_size=(3, 7, 7),
            in_chans=3,
            embed_dim=64,
            stride=(2, 4, 4),
            padding=(1, 3, 3),
            norm_layer=None,
            pre_norm=True
    ):
        super().__init__()
        self.multi_kernel = len(patch_size) > 3
        assert len(patch_size) == len(stride) == len(padding)
        assert len(patch_size) % 3 == 0
        self.patch_size = patch_size

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size[:3], stride=stride[:3], padding=padding[:3])
        if self.multi_kernel:
            kernel_num = len(patch_size) // 3
            self.proj_other = nn.Sequential(
                *[nn.Sequential(
                    nn.GroupNorm(1, embed_dim),
                    nn.GELU(),
                    nn.Conv3d(embed_dim, embed_dim, kernel_size=patch_size[3 * i:3 * (i + 1)],
                              stride=stride[3 * i:3 * (i + 1)], padding=padding[3 * i:3 * (i + 1)])
                ) for i in range(1, kernel_num)]
            )

        dim_norm = in_chans if pre_norm else embed_dim
        self.norm = norm_layer(dim_norm) if norm_layer else None

        self.pre_norm = pre_norm

    def forward(self, x, size):
        D, H, W = size
        if len(x.size()) == 3:
            if self.norm and self.pre_norm:
                x = self.norm(x)
            x = rearrange(x, 'b (d h w) c -> b c d h w', h=H, w=W)

        x = self.proj(x)
        if self.multi_kernel:
            x = self.proj_other(x)

        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        if self.norm and not self.pre_norm:
            x = self.norm(x)

        return x, (D, H, W)


class ChannelAttention(nn.Module):
    def __init__(self, dim, groups=8, qkv_bias=False):
        super().__init__()

        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (three groups group_c) -> three b groups n group_c', three=3, groups=self.groups)
        q, k, v = qkv.unbind(0)
        q = q * (N ** -0.5)
        attention = q.transpose(-1, -2) @ k  # [b groups group_c group_c]
        attention = attention.softmax(dim=-1)
        x = attention @ v.transpose(-1, -2)  # [b groups group_c n]

        x = rearrange(x, 'b groups group_c n -> b n (groups group_c)')

        x = self.proj(x)
        return x


class ChannelBlock(nn.Module):

    def __init__(self, dim, groups, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False, init_value=None):
        super().__init__()

        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3, act=cpe_act),
                                  ConvPosEnc(dim=dim, k=3, act=cpe_act)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, groups=groups, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True) \
            if init_value is not None else None
        self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True) \
            if init_value is not None and self.ffn else None

    def forward(self, x, size):
        x = self.cpe[0](x, size)
        cur = self.norm1(x)
        cur = self.attn(cur)

        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * cur)
        else:
            x = x + self.drop_path(cur)

        x = self.cpe[1](x, size)
        if self.ffn:
            if self.gamma_2 is not None:
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size


def window_partition(x, window_size, _type='xy'):
    assert _type in ['xy', 'tx', 'ty', 'txy', 't']
    B, D, H, W, C = x.shape
    if _type == 'txy':
        windows = rearrange(x, 'b (d win_d) (h win_h) (w win_w) c -> (b d h w) (win_d win_h win_w) c',
                            win_d=window_size[0], win_h=window_size[1], win_w=window_size[2])
    elif _type == 't':
        windows = rearrange(x, 'b (d win_d) H W c -> (b d H W) win_d c',
                            win_d=window_size[0])
    elif _type == 'xy':
        windows = rearrange(x, 'b D (h win_h) (w win_w) c -> (b D h w) (win_h win_w) c',
                            win_h=window_size[1], win_w=window_size[2])
    elif _type == 'tx':
        windows = rearrange(x, 'b (d win_d) H (w win_w) c -> (b d H w) (win_d win_w) c',
                            win_d=window_size[0], win_w=window_size[2])
    else:  # ty
        windows = rearrange(x, 'b (d win_d) (h win_h) W c -> (b d h W) (win_d win_h) c',
                            win_d=window_size[0], win_h=window_size[1])

    return windows


def window_reverse(windows, window_size, D: int, H: int, W: int, _type='xy'):
    assert _type in ['xy', 'tx', 'ty', 'txy', 't']
    d, h, w = D // window_size[0], H // window_size[1], W // window_size[2]
    if _type == 'txy':
        x = rearrange(windows, '(b d h w) (win_d win_h win_w) c -> b (d win_d) (h win_h) (w win_w) c',
                      d=d, h=h, w=w, win_d=window_size[0], win_h=window_size[1], win_w=window_size[2])
    elif _type == 't':
        x = rearrange(windows, '(b d H W) win_d c -> b (d win_d) H W c',
                      d=d, H=H, W=W)
    elif _type == 'xy':
        x = rearrange(windows, '(b D h w) (win_h win_w) c -> b D (h win_h) (w win_w) c',
                      D=D, h=h, w=w, win_h=window_size[1], win_w=window_size[2])
    elif _type == 'tx':
        x = rearrange(windows, '(b d H w) (win_d win_w) c -> b (d win_d) H (w win_w) c',
                      d=d, H=H, w=w, win_d=window_size[0], win_w=window_size[2])
    else:  # ty
        x = rearrange(windows, '(b d h W) (win_d win_h) c -> b (d win_d) (h win_h) W c',
                      d=d, h=h, W=W, win_d=window_size[0], win_h=window_size[1])
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_th = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_tw = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_th = nn.Linear(dim, dim)
        self.proj_tw = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def _attention(self, qkv):
        qkv = rearrange(qkv, 'b n (three num_heads head_c) -> three b num_heads n head_c',
                        three=3, num_heads=self.num_heads)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        x = (attn @ v)
        x = rearrange(x, 'b num_heads n head_c -> b n (num_heads head_c)')
        return x

    def forward(self, x, size):
        D, H, W = size
        B, N, C = x.shape
        assert N == D * H * W

        x = rearrange(x, 'b (d h w) c -> b d h w c', d=D, h=H)
        x_th = x
        x_tw = x

        # window partition
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, 0, 0))
        x_th = F.pad(x_th, (0, 0, 0, 0, pad_t, pad_b, pad_d0, pad_d1))
        x_tw = F.pad(x_tw, (0, 0, pad_l, pad_r, 0, 0, pad_d0, pad_d1))

        _, _, Hp, Wp, _ = x.shape
        _, Dp, _, _, _ = x_th.shape

        x = window_partition(x, self.window_size, _type='xy')
        x_th = window_partition(x_th, self.window_size, _type='ty')
        x_tw = window_partition(x_tw, self.window_size, _type='tx')

        # Windowed self-attention
        qkv = self.qkv(x)
        x = self._attention(qkv)
        x = self.proj(x)

        qkv_th = self.qkv_th(x_th)
        x_th = self._attention(qkv_th)
        x_th = self.proj_th(x_th)

        qkv_tw = self.qkv_tw(x_tw)
        x_tw = self._attention(qkv_tw)
        x_tw = self.proj_tw(x_tw)

        # merge_window
        x = window_reverse(x, self.window_size, D, Hp, Wp, _type='xy')
        x_th = window_reverse(x_th, self.window_size, Dp, Hp, W, _type='ty')
        x_tw = window_reverse(x_tw, self.window_size, Dp, H, Wp, _type='tx')

        if pad_r + pad_b > 0:
            x = x[:, :, :H, :W, :].contiguous()
        if pad_b + pad_d1 > 0:
            x_th = x_th[:, :D, :H, :, :].contiguous()
        if pad_r + pad_d1 > 0:
            x_tw = x_tw[:, :D, :, :W, :].contiguous()

        x = x + x_th + x_tw
        x = rearrange(x, 'b d h w c -> b (d h w) c')

        return x


class WindowAttentionT2DSShared(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, fusion_method='alpha', **kwargs):

        super().__init__()
        assert fusion_method in ['alpha', 'sum', 'weighted_sum', 'sigmoid_alpha']
        self.fusion_method = fusion_method
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_t = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_t = nn.Linear(dim, dim)

        if self.fusion_method in ['alpha', 'sigmoid_alpha']:
            self.alpha = nn.Parameter(1e-4 * torch.ones((1, 1, 1, 1, dim)), requires_grad=True)
        if self.fusion_method == 'weighted_sum':
            self.reweight = Mlp(dim, int(dim * 0.5), dim * 2)

        self.softmax = nn.Softmax(dim=-1)

    def _attention(self, qkv):
        qkv = rearrange(qkv, 'b n (three num_heads head_c) -> three b num_heads n head_c',
                        three=3, num_heads=self.num_heads)
        q, k, v = qkv.unbind(0)
        with torch.cuda.amp.autocast(enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = self.softmax(attn)
            x = (attn @ v)
        x = rearrange(x, 'b num_heads n head_c -> b n (num_heads head_c)')
        return x

    def forward(self, x, size):
        D, H, W = size
        B, N, C = x.shape
        assert N == D * H * W

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]

        # XY
        x = rearrange(x, 'b (d h w) c -> b d h w c', d=D, h=H)
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, 0, 0))
        _, _, Hp, Wp, _ = x.shape
        x = window_partition(x, self.window_size, _type='xy')
        qkv = self.qkv(x)
        x = self._attention(qkv)
        x = self.proj(x)
        x = window_reverse(x, self.window_size, D, Hp, Wp, _type='xy')
        if pad_r + pad_b > 0:
            x = x[:, :, :H, :W, :].contiguous()

        # T
        x_t = x
        qkv_t = self.qkv_t(x_t)

        qkv_th = F.pad(qkv_t, (0, 0, 0, 0, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, _, _ = qkv_th.shape
        qkv_th = window_partition(qkv_th, self.window_size, _type='ty')
        x_th = self._attention(qkv_th)
        x_th = window_reverse(x_th, self.window_size, Dp, Hp, W, _type='ty')
        if pad_b + pad_d1 > 0:
            x_th = x_th[:, :D, :H, :, :].contiguous()

        qkv_tw = F.pad(qkv_t, (0, 0, pad_l, pad_r, 0, 0, pad_d0, pad_d1))
        _, Dp, _, Wp, _ = qkv_tw.shape
        qkv_tw = window_partition(qkv_tw, self.window_size, _type='tx')
        x_tw = self._attention(qkv_tw)
        x_tw = window_reverse(x_tw, self.window_size, Dp, H, Wp, _type='tx')
        if pad_r + pad_d1 > 0:
            x_tw = x_tw[:, :D, :, :W, :].contiguous()

        x_t = self.proj_t(x_th + x_tw)

        x = self._fusion(x, x_t)

        x = rearrange(x, 'b d h w c -> b (d h w) c')

        return x

    def _fusion(self, x, x_t):
        if self.fusion_method == 'alpha':
            x = x + self.alpha * x_t
        elif self.fusion_method == 'sigmoid_alpha':
            x = x + torch.sigmoid(self.alpha) * x_t
        elif self.fusion_method == 'weighted_sum':
            a = (x + x_t).mean(dim=[1, 2, 3])
            a = self.reweight(a)
            a = rearrange(a, 'b (two c) -> two b c', two=2).softmax(dim=0)
            a = rearrange(a, 'two b c -> two b 1 1 1 c').unbind(0)
            x = a[0] * x + a[1] * x_t
        else:
            x = x + x_t
        return x


class WindowAttention2P1D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_t = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_t = nn.Linear(dim, dim)

        self.alpha = nn.Parameter(1e-4 * torch.ones((1, 1, 1, 1, dim)), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def _attention(self, qkv):
        qkv = rearrange(qkv, 'b n (three num_heads head_c) -> three b num_heads n head_c',
                        three=3, num_heads=self.num_heads)
        q, k, v = qkv.unbind(0)
        with torch.cuda.amp.autocast(enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = self.softmax(attn)
            x = (attn @ v)
        x = rearrange(x, 'b num_heads n head_c -> b n (num_heads head_c)')
        return x

    def forward(self, x, size):
        D, H, W = size
        B, N, C = x.shape
        assert N == D * H * W

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]

        # XY
        x = rearrange(x, 'b (d h w) c -> b d h w c', d=D, h=H)
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, 0, 0))
        _, _, Hp, Wp, _ = x.shape
        x = window_partition(x, self.window_size, _type='xy')
        qkv = self.qkv(x)
        x = self._attention(qkv)
        x = self.proj(x)
        x = window_reverse(x, self.window_size, D, Hp, Wp, _type='xy')
        if pad_r + pad_b > 0:
            x = x[:, :, :H, :W, :].contiguous()

        # T
        x_t = x
        qkv_t = self.qkv_t(x_t)

        qkv_t = F.pad(qkv_t, (0, 0, 0, 0, 0, 0, pad_d0, pad_d1))
        _, Dp, _, _, _ = qkv_t.shape
        qkv_t = window_partition(qkv_t, self.window_size, _type='t')
        x_t = self._attention(qkv_t)
        x_t = window_reverse(x_t, self.window_size, Dp, H, W, _type='t')
        if pad_d1 > 0:
            x_t = x_t[:, :D, :, :, :].contiguous()

        x_t = self.proj_t(x_t)
        x = x + self.alpha * x_t

        x = rearrange(x, 'b d h w c -> b (d h w) c')

        return x


class WindowAttention2D(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def _attention(self, qkv):
        qkv = rearrange(qkv, 'b n (three num_heads head_c) -> three b num_heads n head_c',
                        three=3, num_heads=self.num_heads)
        q, k, v = qkv.unbind(0)
        with torch.cuda.amp.autocast(enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = self.softmax(attn)
            x = (attn @ v)
        x = rearrange(x, 'b num_heads n head_c -> b n (num_heads head_c)')
        return x

    def forward(self, x, size):
        D, H, W = size
        B, N, C = x.shape
        assert N == D * H * W

        x = rearrange(x, 'b (d h w) c -> b d h w c', d=D, h=H)

        # window partition
        pad_l = pad_t = 0
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, 0, 0))
        _, _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size, _type='xy')

        # Windowed self-attention
        qkv = self.qkv(x)
        x = self._attention(qkv)
        x = self.proj(x)

        # merge_window
        x = window_reverse(x, self.window_size, D, Hp, Wp, _type='xy')

        if pad_r + pad_b > 0:
            x = x[:, :, :H, :W, :].contiguous()

        x = rearrange(x, 'b d h w c -> b (d h w) c')

        return x


class WindowAttention3D(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def _attention(self, qkv):
        qkv = rearrange(qkv, 'b n (three num_heads head_c) -> three b num_heads n head_c',
                        three=3, num_heads=self.num_heads)
        q, k, v = qkv.unbind(0)
        # with torch.cuda.amp.autocast(enabled=False):
        #     q, k, v = q.float(), k.float(), v.float()
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        x = (attn @ v)
        x = rearrange(x, 'b num_heads n head_c -> b n (num_heads head_c)')
        return x

    def forward(self, x, size):
        D, H, W = size
        B, N, C = x.shape
        assert N == D * H * W

        x = rearrange(x, 'b (d h w) c -> b d h w c', d=D, h=H)

        # window partition
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))

        _, Dp, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size, _type='txy')

        # Windowed self-attention
        qkv = self.qkv(x)
        x = self._attention(qkv)
        x = self.proj(x)

        # merge_window
        x = window_reverse(x, self.window_size, Dp, Hp, Wp, _type='txy')

        if pad_r + pad_b + pad_d1 > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        x = rearrange(x, 'b d h w c -> b (d h w) c')

        return x


class SpatialBlock(nn.Module):
    r""" Same as the Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(8, 7, 7),
                 mlp_ratio=4., qkv_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False, init_value=None,
                 attn_type='t2d', **kwargs):
        super().__init__()
        assert attn_type in ['t2d', '2d', '3d', 't2ds', 't2dss', '2p1d']
        if attn_type == 't2d':
            attn_func = WindowAttention
        elif attn_type == '2d':
            attn_func = WindowAttention2D
        elif attn_type == 't2ds':
            attn_func = WindowAttentionT2DS
        elif attn_type == 't2dss':
            attn_func = WindowAttentionT2DSShared
        elif attn_type == '2p1d':
            attn_func = WindowAttention2P1D
        else:
            attn_func = WindowAttention3D
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3, act=cpe_act),
                                  ConvPosEnc(dim=dim, k=3, act=cpe_act)])

        self.norm1 = norm_layer(dim)
        self.attn = attn_func(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias, **kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

        self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True) \
            if init_value is not None else None
        self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True) \
            if init_value is not None and self.ffn else None

    def forward(self, x, size):
        shortcut = self.cpe[0](x, size)
        x = self.norm1(shortcut)
        x = self.attn(x, size)

        if self.gamma_1 is not None:
            x = shortcut + self.drop_path(self.gamma_1 * x)
        else:
            x = shortcut + self.drop_path(x)

        x = self.cpe[1](x, size)
        if self.ffn:
            if self.gamma_2 is not None:
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size

