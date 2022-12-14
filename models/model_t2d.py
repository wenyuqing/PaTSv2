import os
from collections import OrderedDict
from tqdm import tqdm

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange, repeat
import clip_custom
from .prompt import VideoSpecificPrompt

from .k400_class import K400_CLASS


SINGLE_PROMPT = ["a video of a person {}."]


class CLIPTextClassifier(nn.Module):
    def __init__(self, pretrain, backbone_name, num_classes=400, bias=False, embed_dim=512, vpt=False):
        super(CLIPTextClassifier, self).__init__()

        text_features, logit_scale = self.get_text_features(pretrain, backbone_name)
        self.logit_scale = nn.Parameter(logit_scale)
        self.register_buffer('text_features', text_features)
        self.bias = None
        self.prompts_t = None
        if vpt:
            self.prompts_t = VideoSpecificPrompt(layers=2, embed_dim=embed_dim, alpha=0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x, images=None):
        x = x / x.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        text_features = self.text_features.unsqueeze(0).expand(x.shape[0], -1, -1)
        if self.prompts_t is not None:
            images = images.mean(dim=1, keepdim=False)
            text_features = text_features + self.prompts_t(text_features.transpose(1, 2), images).transpose(1, 2)  # B C n_class
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
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
        clip_model = clip_custom.load(pretrain)[0]
        clip_model = clip_model.to(torch.float32)
        text_encoder = clip_model.encode_text
        tokenizer = lambda x: clip_custom.tokenize(x).cuda()
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
                 attn_type='2d',
                 use_cls_token=False):
        super(Attention, self).__init__()
        assert attn_type in ['3d', '2d', 't2d', '2p1d', 't2dfs', '2dp1d', 't2dns']
        self.use_cls_token = use_cls_token
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_type = attn_type
        if self.attn_type in ['t2d', '2p1d', '2dp1d']:
            self.in_proj_weight_t = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
            self.in_proj_bias_t = nn.Parameter(torch.empty(3 * embed_dim))
            self.out_proj_t = nn.Linear(embed_dim, embed_dim)
        if self.attn_type in ['t2dns']:
            self.in_proj_weight_th = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
            self.in_proj_bias_th = nn.Parameter(torch.empty(3 * embed_dim))
            self.out_proj_th = nn.Linear(embed_dim, embed_dim)

            self.in_proj_weight_tw = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
            self.in_proj_bias_tw = nn.Parameter(torch.empty(3 * embed_dim))
            self.out_proj_tw = nn.Linear(embed_dim, embed_dim)

        if self.attn_type in ['t2d', '2p1d', 't2dfs', '2dp1d', 't2dns']:
            self.alpha = nn.Parameter(1e-4 * torch.ones((embed_dim)), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

        if self.attn_type in ['t2d', '2p1d', '2dp1d']:
            xavier_uniform_(self.in_proj_weight_t)
            constant_(self.in_proj_bias_t, 0.)
            constant_(self.out_proj_t.bias, 0.)
        if self.attn_type in ['t2dns']:
            xavier_uniform_(self.in_proj_weight_th)
            constant_(self.in_proj_bias_th, 0.)
            constant_(self.out_proj_th.bias, 0.)

            xavier_uniform_(self.in_proj_weight_tw)
            constant_(self.in_proj_bias_tw, 0.)
            constant_(self.out_proj_tw.bias, 0.)


    def attention(self, qkv):
        qkv = rearrange(qkv, 'b n (three num_heads head_c) -> three b num_heads n head_c',
                        three=3, num_heads=self.num_heads)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(-1)
        x = (attn @ v)
        x = rearrange(x, 'b num_heads n head_c -> b n (num_heads head_c)')
        return x

    def _forward_3d(self, x):
        x = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        x = self.attention(x)
        x = self.out_proj(x)
        return x

    def _forward_2d(self, x, size):
        T, H, W = size

        x = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        x = rearrange(x, 'b (t hw) c -> (b t) hw c', t=T)
        x = self.attention(x)
        x = rearrange(x, '(b t) hw c -> b (t hw) c', t=T)
        x = self.out_proj(x)
        return x

    def _forward_t2d(self, x, size):
        T, H, W = size

        x = self._forward_2d(x, size)

        if self.use_cls_token:
            x_t = rearrange(x, 'b (t hw) c -> b t hw c', t=T)
            x_cls = x_t[:, :, 0:1, :]
            x_t = x_t[:, :, 1:, :]
            x_t = F.linear(x_t, self.in_proj_weight_t, self.in_proj_bias_t)


            x_th = rearrange(x_t, 'b t (h w) c -> (b w) (t h) c', t=T, h=H)
            x_th = self.attention(x_th)
            x_th = rearrange(x_th, '(b w) (t h) c -> b t (h w) c', t=T, w=W)

            x_tw = rearrange(x_t, 'b t (h w) c -> (b h) (t w) c', t=T, h=H)
            x_tw = self.attention(x_tw)
            x_tw = rearrange(x_tw, '(b h) (t w) c -> b t (h w) c', t=T, h=H)

            x_t = x_th + x_tw
            x_t = torch.cat([x_cls, x_t], dim=2)
            x_t = rearrange(x_t, 'b t hw c -> b (t hw) c')
            x_t = self.out_proj_t(x_t)
        else:
            x_t = F.linear(x, self.in_proj_weight_t, self.in_proj_bias_t)

            x_th = rearrange(x_t, 'b (t h w) c -> (b w) (t h) c', t=T, h=H)
            x_th = self.attention(x_th)
            x_th = rearrange(x_th, '(b w) (t h) c -> b (t h w) c', t=T, w=W)

            x_tw = rearrange(x_t, 'b (t h w) c -> (b h) (t w) c', t=T, h=H)
            x_tw = self.attention(x_tw)
            x_tw = rearrange(x_tw, '(b h) (t w) c -> b (t h w) c', t=T, h=H)

            x_t = x_th + x_tw
            x_t = self.out_proj_t(x_t)

        x = x + x_t * self.alpha
        return x

    def _forward_2dp1d(self, x, size):
        T, H, W = size

        x = self._forward_2d(x, size)

        x_t = F.linear(x, self.in_proj_weight_t, self.in_proj_bias_t)

        x_t = rearrange(x_t, 'b (t h w) c -> (b h w) t c', h=H, w=W)
        x_t = self.attention(x_t)
        x_t = rearrange(x_t, '(b h w) t c -> b (t h w) c', h=H, w=W)

        x_t = self.out_proj_t(x_t)

        x = x + x_t * self.alpha
        return x

    def _forward_t2dfs(self, x, size):
        T, H, W = size

        x = F.linear(x, self.in_proj_weight, self.in_proj_bias)

        x_hw = rearrange(x, 'b (t hw) c -> (b t) hw c', t=T)
        x_hw = self.attention(x_hw)
        x_hw = rearrange(x_hw, '(b t) hw c -> b (t hw) c', t=T)

        x_th = rearrange(x, 'b (t h w) c -> (b w) (t h) c', t=T, h=H)
        x_th = self.attention(x_th)
        x_th = rearrange(x_th, '(b w) (t h) c -> b (t h w) c', t=T, w=W)

        x_tw = rearrange(x, 'b (t h w) c -> (b h) (t w) c', t=T, h=H)
        x_tw = self.attention(x_tw)
        x_tw = rearrange(x_tw, '(b h) (t w) c -> b (t h w) c', t=T, h=H)

        x = x_hw + self.alpha * (x_tw + x_th)
        x = self.out_proj(x)

        return x

    def _forward_t2dns(self, x, size):
        T, H, W = size

        x = self._forward_2d(x, size)

        x_th = F.linear(x, self.in_proj_weight_th, self.in_proj_bias_th)
        x_tw = F.linear(x, self.in_proj_weight_tw, self.in_proj_bias_tw)

        x_th = rearrange(x_th, 'b (t h w) c -> (b w) (t h) c', t=T, h=H)
        x_th = self.attention(x_th)
        x_th = rearrange(x_th, '(b w) (t h) c -> b (t h w) c', t=T, w=W)

        x_tw = rearrange(x_tw, 'b (t h w) c -> (b h) (t w) c', t=T, h=H)
        x_tw = self.attention(x_tw)
        x_tw = rearrange(x_tw, '(b h) (t w) c -> b (t h w) c', t=T, h=H)

        x_th = self.out_proj_th(x_th)
        x_tw = self.out_proj_tw(x_tw)

        x = x + (x_th + x_tw) * self.alpha
        return x

    def forward(self, x, size):
        """[B N C]"""
        if self.attn_type == '3d':
            return self._forward_3d(x)

        if self.attn_type == '2d':
            return self._forward_2d(x, size)

        if self.attn_type == 't2d':
            return self._forward_t2d(x, size)

        if self.attn_type == 't2dfs':
            return self._forward_t2dfs(x, size)

        if self.attn_type == '2dp1d':
            return self._forward_2dp1d(x, size)

        if self.attn_type == 't2dns':
            return self._forward_t2dns(x, size)
        raise NotImplementedError('attention type: {} is not implemented'.format(self.attn_type))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_type='2d', use_cls_token=False):
        super().__init__()

        self.attn = Attention(d_model, n_head, attn_type, use_cls_token)
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

class ResidualAttentionBlockTSF(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = Attention(d_model, n_head, '2d')
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

        self.temporal_norm = LayerNorm(d_model)
        self.temporal_attn = Attention(d_model, n_head, '3d')
        self.temporal_fc = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.temporal_fc.bias, 0.)
        constant_(self.temporal_fc.weight, 0.)

    def forward(self, x: torch.Tensor, size):
        T, H, W = size
        xt = rearrange(x, 'b (t h w) c -> (b h w) t c', h=H, w=W)
        res_temporal = self.temporal_attn(self.temporal_norm(xt), size)
        res_temporal = rearrange(res_temporal, '(b h w) t c -> b (t h w) c', h=H, w=W)
        res_temporal = self.temporal_fc(res_temporal)
        xt = x + res_temporal

        xs = xt
        res_spatial = self.attn(self.ln_1(xs), size)
        x = xt + res_spatial

        x = x + self.mlp(self.ln_2(x))
        return x, size


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_type='2d', enable_checkpoint=False, use_cls_token=False):
        super().__init__()
        self.width = width
        self.layers = layers
        if attn_type in ['tsf']:
            self.resblocks = MySequential(*[ResidualAttentionBlockTSF(width, heads) for _ in range(layers)])
        else:
            self.resblocks = MySequential(*[ResidualAttentionBlock(width, heads, attn_type, use_cls_token) for _ in range(layers)])
        self.enable_checkpoint = enable_checkpoint

    def forward(self, x: torch.Tensor, size):
        checkpoint_segment = 2
        if self.enable_checkpoint:
            # FIXME: use segment checkpointing to accelerate training
            # for block in self.resblocks:
            #     x, size = checkpoint.checkpoint(block, x, size)
            x, size = sequential_checkpoint(self.resblocks, checkpoint_segment, x, size)
            return x
        else:
            return self.resblocks(x, size)[0]

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

class CLIPT2D(nn.Module):
    def __init__(self,
                 num_classes=400,
                 width=768,
                 patch_size=(2, 16, 16),
                 layers=12,
                 heads=12,
                 input_resolution=224,
                 frames=32,
                 pretrain=None,
                 attn_type='t2d',
                 pos_type='sep',
                 model_type='3d',
                 temporal_model='avg',
                 use_cls_token=False,
                 temporal_layer=4,
                 enable_checkpoint=False,
                 use_text_classifier=True,
                 text_dim=512,
                 text_heads=8,
                 text_backbone_name='ViT-B/16',
                 text_bias=False,
                 vpt=False,
                 use_temporal_cls_token=False,
                 imagenet_pretrain=None,
                 pretrain_jit=True
                 ):
        # input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int
        super().__init__()
        # parse config
        freeze = False

        if use_cls_token:
            assert attn_type in ['2d', '3d', 't2d'], f'attention {attn_type} do not support class token'
            # assert pos_type == 'spatial', 'only support spatial position embedding with cls token'
        # build model
        self.attn_type = attn_type
        self.pos_type = pos_type
        self.input_resolution = input_resolution
        self.frames = frames
        self.model_type = model_type
        self.use_cls_token = use_cls_token
        self.enable_checkpoint = enable_checkpoint
        self.use_text_classifier = use_text_classifier
        self.use_temporal_cls_token = use_temporal_cls_token

        if model_type == '3d':
            self.conv1 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        else:
            assert pos_type == 'spatial'
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size[1:], stride=patch_size[1:],
                                   bias=False)
        scale = width ** -0.5
        if self.use_cls_token:
            self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size[1]) ** 2 +
                                                                     (1 if self.use_cls_token else 0), width))
        if pos_type not in ['spatial']:
            self.pos_t = nn.Parameter(scale * torch.randn((frames // patch_size[0]), width))

        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, attn_type, enable_checkpoint, use_cls_token)
        self.ln_post = LayerNorm(width)

        # vpt
        self.vpt = vpt
        if not self.use_text_classifier:
            assert not self.vpt

        if self.use_text_classifier:
            self.proj = nn.Parameter(torch.randn(width, text_dim))
            self.head = CLIPTextClassifier(pretrain, text_backbone_name, num_classes, bias=text_bias, embed_dim=text_dim,
                                           vpt=vpt)
            if self.vpt:
                self.prompts_visual_ln = LayerNorm(width)
                self.prompts_visual_proj = nn.Parameter(torch.randn(width, text_dim))
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
            self.model_t_pos = nn.Parameter(scale * torch.randn((frames // patch_size[0]) +
                                                                (1 if self.use_temporal_cls_token else 0), t_width))
            self.model_t = Transformer(t_width,
                                       temporal_layer,
                                       t_head, '3d')
            if self.use_temporal_cls_token:
                self.class_embedding_t = nn.Parameter(scale * torch.randn(t_width))

        self._initialize_weights()
        self._load_pretrain(pretrain, jit=pretrain_jit)
        self._load_in_pretrain(imagenet_pretrain)

        self.freeze = freeze

    def _initialize_weights(self):
        nn.init.normal_(self.positional_embedding, std=0.02)
        if self.pos_type not in ['spatial']:
            nn.init.normal_(self.pos_t, std=0.02)
        if self.use_cls_token:
            nn.init.normal_(self.class_embedding, std=0.02)
        if self.temporal_model == 'transformer':
            nn.init.normal_(self.model_t_pos, std=0.02)
            if self.use_temporal_cls_token:
                nn.init.normal_(self.class_embedding_t, std=0.02)

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

    def _load_pretrain(self, pretrain, jit=True):
        if pretrain is None or not os.path.exists(pretrain):
            return
        print(f'Loading network weights from {pretrain}')
        model_state_dict_3d = self.state_dict()

        if jit:
            clip_model = torch.jit.load(pretrain, map_location='cpu')
            clip_model = clip_model.visual
            model_state_dict_2d = clip_model.state_dict()
        else:
            clip_model = torch.load(pretrain, map_location='cpu')
            model_state_dict_2d = {k.replace('visual.', ''): v for k, v in clip_model.items() if k.startswith('visual')}

        # remove pos embed for class token
        if not self.use_cls_token:
            model_state_dict_2d['positional_embedding'] = model_state_dict_2d['positional_embedding'][1:]

        inflated_model_dict = inflate_weight(model_state_dict_2d, model_state_dict_3d)
        msg = self.load_state_dict(inflated_model_dict, strict=False)
        print(msg)
        print('Pretrained network weights loaded.')

    def forward_features(self, x: torch.Tensor):
        B = x.shape[0]
        if self.model_type == '2d':
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = self.conv1(x)
            size = [x.shape[0] // B] + list(x.shape[2:])
            x = rearrange(x, '(b t) c h w -> b (t h w) c', b=B)
        else:
            x = self.conv1(x)
            size = x.shape[2:]
            x = rearrange(x, 'b c t h w -> b (t h w) c')

        if self.use_cls_token:
            x = rearrange(x, 'b (t hw) c -> (b t) hw c', t=size[0])
            x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = rearrange(x, '(b t) hw c -> b (t hw) c', t=size[0])

        if self.pos_type == 'sep':
            pos_embed = (rearrange(self.positional_embedding, 'hw c -> 1 hw c') +
                         rearrange(self.pos_t, 't c -> t 1 c')).view(-1, x.shape[-1])
        elif self.pos_type == 'spatial':
            pos_embed = repeat(self.positional_embedding, 'hw c -> t hw c', t=size[0]).reshape(-1, x.shape[-1])
        else:
            raise NotImplementedError
        x = x + pos_embed

        x = self.ln_pre(x)
        x = self.transformer(x, size)
        x = rearrange(x, 'b (t hw) c -> b t hw c', t=size[0])

        img_features = x
        if self.vpt:
            if self.use_cls_token:
                img_features = img_features[:, :, 0]
            img_features = self.prompts_visual_ln(img_features)
            img_features = img_features @ self.prompts_visual_proj

        if self.use_cls_token:
            x = self.ln_post(x[:, :, 0])
        else:
            x = self.ln_post(x.mean(dim=2))

        if self.use_text_classifier:
            x = x @ self.proj
        return x, img_features

    def forward(self, x, text):
        # b t c h w -> b c t h w
        x = x.permute(0, 2, 1, 3, 4)
        x, img_features = self.forward_features(x)
        if self.temporal_model == 'avg':
            x = x.mean(dim=1)
        elif self.temporal_model == 'transformer':
            if self.use_temporal_cls_token:
                x = torch.cat([self.class_embedding_t + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),  x], dim=1)
                x = x + self.model_t_pos
                x = self.model_t(x, None)
                x = x[:, 0]
            else:
                orig_x = x
                x = x + self.model_t_pos
                x = self.model_t(x, None)
                x = x + orig_x
                x = x.mean(dim=1)
        else:
            raise NotImplementedError
        if self.use_text_classifier:
            x = self.head(x, img_features)
        else:
            x = self.head(x)
        return x