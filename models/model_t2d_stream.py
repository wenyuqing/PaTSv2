import os
from collections import OrderedDict
from tqdm import tqdm
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange, repeat
import clip
from .prompt import VideoSpecificPrompt

from .k400_class import K400_CLASS


SINGLE_PROMPT = ["a video of a person {}."]


class CLIPTextClassifier(nn.Module):
    def __init__(self, pretrain, backbone_name, num_classes=400, bias=False, embed_dim=512,):
        super(CLIPTextClassifier, self).__init__()

        text_features, logit_scale = self.get_text_features(pretrain, backbone_name)
        self.logit_scale = nn.Parameter(logit_scale)
        self.register_buffer('text_features', text_features)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        x = x / x.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        text_features = self.text_features.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = logit_scale * (x.unsqueeze(1) @ text_features).squeeze(1)  # [B, C] @ [B, C, n_class] -> [B, n_class]
        if self.bias is not None:
            x = x + self.bias
        if not self.training:
            x = F.softmax(x, dim=1)
        return x

    @staticmethod
    def get_text_features(pretrain, backbone_name='ViT-B/16'):
        prompts = SINGLE_PROMPT
        classnames = [_[0] for _ in sorted(K400_CLASS, key=lambda x: x[1])]
        clip_model = clip.load(backbone_name, download_root=os.path.dirname(pretrain))[0]
        clip_model = clip_model.to(torch.float32)
        text_encoder = clip_model.encode_text
        tokenizer = lambda x: clip.tokenize(x).cuda()
        texts_weights = []
        for classname in tqdm(classnames):
            texts = [prompt.format(classname).lower() for prompt in prompts]
            texts = tokenizer(texts)
            with torch.no_grad():
                class_embeddings = text_encoder(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            texts_weights.append(class_embedding)
        text_features = torch.stack(texts_weights, dim=1)
        text_features = text_features / text_features.norm(dim=0, keepdim=True)
        return text_features, clip_model.state_dict()['logit_scale']


class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            x = self.act(x)
        return x


def inflate_weight(state_dict_2d, state_dict_3d):
    # copy from slowfast.checkpoint
    from collections import OrderedDict
    state_dict_inflated = OrderedDict()
    for k, v2d in state_dict_2d.items():
        if k not in state_dict_3d.keys():
            print(f"Unknown key {k} from 2d dict")
            continue
        v3d = state_dict_3d[k]
        # Inflate the weight of 2D conv to 3D conv.
        if len(v2d.shape) == 4 and len(v3d.shape) == 5:
            print(
                "Inflate {}: {} -> {}: {}".format(k, v2d.shape, k, v3d.shape)
            )
            # Dimension need to be match.
            assert v2d.shape[-2:] == v3d.shape[-2:]
            assert v2d.shape[:2] == v3d.shape[:2]
            v3d = (
                    v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
            )
        elif v2d.shape == v3d.shape:
            v3d = v2d
        else:
            print(
                "Unexpected {}: {} -|> {}: {}".format(
                    k, v2d.shape, k, v3d.shape
                )
            )
        state_dict_inflated[k] = v3d.clone()
    return state_dict_inflated


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_type='full'):
        super(Attention, self).__init__()
        assert attn_type in ['full', 'stream', 'stream_batch']
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if attn_type in ['stream', 'stream_batch']:
            self.in_proj_weight_t = nn.Parameter(torch.empty((embed_dim, embed_dim)))
            self.in_proj_bias_t = nn.Parameter(torch.empty(embed_dim))
            self.out_proj_t = nn.Linear(embed_dim, embed_dim)
            self.alpha = nn.Parameter(1e-4 * torch.ones((embed_dim)), requires_grad=True)

        self.history_k = []
        self.history_v = []
        self._reset_parameters()

    def clear_history(self):
        self.history_k.clear()
        self.history_v.clear()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

        if self.attn_type in ['stream', 'stream_batch']:
            xavier_uniform_(self.in_proj_weight_t)
            constant_(self.in_proj_bias_t, 0.)
            constant_(self.out_proj_t.bias, 0.)

    def attention(self, q, k, v, mask=None):
        attn = (q @ k.transpose(-2, -1))
        if mask is not None:
            attn = attn.masked_fill(mask, -1e3)
        attn = attn.softmax(-1)
        o = (attn @ v)
        return o

    def _forward_in_frame(self, x):
        x = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = rearrange(x, 'b n (three num_heads head_c) -> three b num_heads n head_c',
                        three=3, num_heads=self.num_heads)
        q, k, v = qkv.unbind(0)
        q = q * self.scale

        x = self.attention(q, k, v)

        x = rearrange(x, 'b num_heads n head_c -> b n (num_heads head_c)')
        x = self.out_proj(x)
        if self.attn_type == 'stream':
            self.history_k.append(k.detach())
            self.history_v.append(v.detach())
        return x

    def _forward_cross_frame(self, x, size):
        H, W = size

        q_t = F.linear(x, self.in_proj_weight_t, self.in_proj_bias_t)
        q_t = q_t * self.scale
        q_th = rearrange(q_t, 'b (h w) (num_heads head_c) -> (b w) num_heads h head_c', num_heads=self.num_heads, h=H)
        q_tw = rearrange(q_t, 'b (h w) (num_heads head_c) -> (b h) num_heads w head_c', num_heads=self.num_heads, w=W)

        k_t = torch.stack(self.history_k, dim=1)
        v_t = torch.stack(self.history_v, dim=1)

        k_th = rearrange(k_t, 'b t num_heads (h w) head_c -> (b w) num_heads (t h) head_c', h=H)
        k_tw = rearrange(k_t, 'b t num_heads (h w) head_c -> (b h) num_heads (t w) head_c', w=W)
        v_th = rearrange(v_t, 'b t num_heads (h w) head_c -> (b w) num_heads (t h) head_c', h=H)
        v_tw = rearrange(v_t, 'b t num_heads (h w) head_c -> (b h) num_heads (t w) head_c', w=W)

        o_th = self.attention(q_th, k_th, v_th)
        o_tw = self.attention(q_tw, k_tw, v_tw)

        o_th = rearrange(o_th, '(b w) num_heads h head_c -> b (h w) (num_heads head_c)', w=W)
        o_tw = rearrange(o_tw, '(b h) num_heads w head_c -> b (h w) (num_heads head_c)', h=H)

        x_t = self.out_proj_t(o_th + o_tw)

        return x_t

    def _forward_batch(self, x, size):
        H, W = size
        B = x.shape[0]
        T = x.shape[1] // (H * W)
        xs, ks, vs = self._forward_2d(x, size)
        ks, vs = ks.detach(), vs.detach()

        q_t = F.linear(x, self.in_proj_weight_t, self.in_proj_bias_t)
        q_t = q_t * self.scale
        q_th = rearrange(q_t, 'b (t h w) (num_heads head_c) -> (b w) num_heads (t h) head_c', num_heads=self.num_heads, h=H, w=W)
        q_tw = rearrange(q_t, 'b (t h w) (num_heads head_c) -> (b h) num_heads (t w) head_c', num_heads=self.num_heads, h=H, w=W)

        v_th = rearrange(vs, '(b t) num_heads (h w) head_c -> (b w) num_heads (t h) head_c', h=H, b=B)
        v_tw = rearrange(vs, '(b t) num_heads (h w) head_c -> (b h) num_heads (t w) head_c', w=W, b=B)
        k_th = rearrange(ks, '(b t) num_heads (h w) head_c -> (b w) num_heads (t h) head_c', h=H, b=B)
        k_tw = rearrange(ks, '(b t) num_heads (h w) head_c -> (b h) num_heads (t w) head_c', w=W, b=B)

        mask_th = self.get_t_mask(T, H, device=x.device)
        mask_tw = self.get_t_mask(T, W, device=x.device)

        o_th = self.attention(q_th, k_th, v_th, mask_th)
        o_tw = self.attention(q_tw, k_tw, v_tw, mask_tw)

        o_th = rearrange(o_th, '(b w) num_heads (t h) head_c -> b (t h w) (num_heads head_c)', h=H, b=B)
        o_tw = rearrange(o_tw, '(b h) num_heads (t w) head_c -> b (t h w) (num_heads head_c)', w=W, b=B)
        x_t = self.out_proj_t(o_th + o_tw)

        x = xs + self.alpha * x_t
        return x

    @lru_cache()
    def get_t_mask(self, T, H, device='cpu'):
        mask = torch.tril(torch.ones((T, T), device=device)).view(T, 1, T, 1)
        mask = mask.repeat(1, H, 1, H)
        mask = mask.view(T * H, T * H)
        mask = (1 - mask).bool()
        return mask

    def _forward_2d(self, x, size):
        H, W = size
        B = x.shape[0]

        x = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = rearrange(x, 'b (t h w) (three num_heads head_c) -> three (b t) num_heads (h w) head_c',
                        three=3, num_heads=self.num_heads, h=H, w=W)
        q, k, v = qkv.unbind(0)
        q = q * self.scale

        x = self.attention(q, k, v)

        x = rearrange(x, '(b t) num_heads hw head_c -> b (t hw) (num_heads head_c)', b=B)
        x = self.out_proj(x)
        return x, k, v

    def forward(self, x, size):
        """[B N C]"""
        if self.attn_type == 'full':
            x = self._forward_in_frame(x)
        elif self.attn_type == 'stream_batch':
            x = self._forward_batch(x, size)
        else:
            x_in = self._forward_in_frame(x)
            x_cross = self._forward_cross_frame(x, size)
            x = x_in + self.alpha * x_cross
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_type: str = 'full'):
        super().__init__()

        self.attn = Attention(d_model, n_head, attn_type)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, size):
        x = x + self.attn(self.ln_1(x), size)
        x = x + self.mlp(self.ln_2(x))
        return x, size

    def clear_history(self):
        self.attn.clear_history()


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_type='full', enable_checkpoint=False):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = MySequential(*[ResidualAttentionBlock(width, heads, attn_type) for _ in range(layers)])
        self.enable_checkpoint = enable_checkpoint

    def forward(self, x: torch.Tensor, size):
        checkpoint_segment = 2
        if self.enable_checkpoint:
            x, size = sequential_checkpoint(self.resblocks, checkpoint_segment, x, size)
            return x
        else:
            return self.resblocks(x, size)[0]

    def clear_history(self):
        for layer in self.resblocks:
            layer.clear_history()


def sequential_checkpoint(functions, segments, *input):
    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    def run_function(start, end, functions):
        def forward(*input):
            for j in range(start, end + 1):
                input = functions[j](*input)
            return input
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        input = checkpoint.checkpoint(run_function(start, end, functions), *input)
    input = run_function(end + 1, len(functions) - 1, functions)(*input)
    return input


class CLIPT2DS(nn.Module):
    def __init__(self,
                 num_classes=400,
                 width=768,
                 patch_size=(2, 16, 16),
                 layers=12,
                 heads=12,
                 input_resolution=224,
                 frames=32,
                 pretrain=None,
                 temporal_model='avg',
                 temporal_layer=4,
                 enable_checkpoint=False,
                 use_text_classifier=True,
                 text_dim=512,
                 text_heads=8,
                 text_backbone_name='ViT-B/16',
                 text_bias=False,
                 imagenet_pretrain=None,
                 batch_mode=False,
                 ):
        # input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int
        super().__init__()

        # build model
        self.batch_mode = batch_mode
        self.input_resolution = input_resolution
        self.frames = frames
        self.enable_checkpoint = enable_checkpoint
        self.use_text_classifier = use_text_classifier

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size[1:], stride=patch_size[1:],
                               bias=False)
        scale = width ** -0.5

        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size[1]) ** 2, width))


        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, 'stream_batch' if self.batch_mode else 'stream', enable_checkpoint)
        self.ln_post = LayerNorm(width)

        if self.use_text_classifier:
            self.proj = nn.Parameter(torch.randn(width, text_dim))
            self.head = CLIPTextClassifier(pretrain, text_backbone_name, num_classes, bias=text_bias, embed_dim=text_dim)
        else:
            self.head = TransformerBasicHead(
                width,
                num_classes,
                dropout_rate=0.5,
            )

        # temporal modeling model
        self.temporal_model = temporal_model
        if temporal_model == 'transformer':
            t_width = text_dim if self.use_text_classifier else width
            t_head = text_heads if self.use_text_classifier else heads
            self.model_t_pos = nn.Parameter(scale * torch.randn((frames // patch_size[0]), t_width))
            self.model_t = Transformer(t_width,
                                       temporal_layer,
                                       t_head, 'full')

        self._initialize_weights()
        self._load_pretrain(pretrain)
        self._load_in_pretrain(imagenet_pretrain)

    def _initialize_weights(self):
        nn.init.normal_(self.positional_embedding, std=0.02)
        if self.temporal_model == 'transformer':
            nn.init.normal_(self.model_t_pos, std=0.02)


    def _load_in_pretrain(self, pretrain):
        if pretrain is None or not os.path.exists(pretrain):
            return
        print(f'Loading network weights from {pretrain}')
        checkpoint = torch.load(pretrain, map_location='cpu')
        load_state_dict = checkpoint['model']
        model_state_dict_3d = self.state_dict()

        print('inflate weights')
        load_state_dict = inflate_weight(load_state_dict, model_state_dict_3d)
        msg = self.load_state_dict(load_state_dict, strict=False)
        print(f"resume model: {msg}")

    def _load_pretrain(self, pretrain):
        if pretrain is None or not os.path.exists(pretrain):
            return
        print(f'Loading network weights from {pretrain}')
        model_state_dict_3d = self.state_dict()

        clip_model = torch.jit.load(pretrain, map_location='cpu')
        clip_model = clip_model.visual
        model_state_dict_2d = clip_model.state_dict()

        # remove pos embed for class token
        model_state_dict_2d['positional_embedding'] = model_state_dict_2d['positional_embedding'][1:]

        inflated_model_dict = inflate_weight(model_state_dict_2d, model_state_dict_3d)
        msg = self.load_state_dict(inflated_model_dict, strict=False)
        print(msg)
        print('Pretrained network weights loaded.')

    def forward_features(self, x: torch.Tensor):
        B = x.shape[0]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)
        size = list(x.shape[2:])
        T = x.shape[0] // B
        x = rearrange(x, '(b t) c h w -> t b (h w) c', b=B)

        pos_embed = rearrange(self.positional_embedding, 'hw c -> 1 1 hw c')

        x = x + pos_embed

        x = self.ln_pre(x)
        if self.batch_mode:
            x = rearrange(x, 't b (h w) c -> b (t h w) c', h=size[0])
            x = self.transformer(x, size)
            x = rearrange(x, 'b (t hw) c -> b t hw c', t=T)
            x = x.mean(dim=2)
        else:
            outs = []
            for i in range(T):
                out = self.transformer(x[i], size).mean(dim=1)
                outs.append(out)
            self.transformer.clear_history()
            x = torch.stack(outs, dim=1)  # b t c
        x = self.ln_post(x)

        if self.use_text_classifier:
            x = x @ self.proj
        return x

    def forward(self, x, text):
        # b t c h w -> b c t h w
        x = x.permute(0, 2, 1, 3, 4)
        x = self.forward_features(x)
        if self.temporal_model == 'avg':
            x = x.mean(dim=1)
        elif self.temporal_model == 'transformer':
            orig_x = x
            x = x + self.model_t_pos
            x = self.model_t(x, None)
            x = x + orig_x
            x = x.mean(dim=1)
        else:
            raise NotImplementedError

        x = self.head(x)
        return x