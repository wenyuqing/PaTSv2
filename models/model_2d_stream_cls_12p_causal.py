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

# def inflate_weight(state_dict_2d, state_dict_3d):
#     # copy from slowfast.checkpoint
#     from collections import OrderedDict
#     state_dict_inflated = OrderedDict()
#     state_dict_inflated['class_embedding'] = state_dict_2d['cls_token'].squeeze()
#     state_dict_inflated['positional_embedding'] = state_dict_2d['pos_embed'].squeeze()
#     state_dict_inflated['conv1.weight'] = state_dict_2d['patch_embed.proj.weight']
#     state_dict_inflated['ln_pre.weight'] = state_dict_2d['norm.weight']
#     state_dict_inflated['ln_pre.bias'] = state_dict_2d['norm.bias']
#     for i in range(12):
#         state_dict_inflated['transformer.resblocks.' + str(i) + '.attn.in_proj_weight'] = state_dict_2d[
#             'blocks.' + str(i) + '.attn.qkv.weight']
#         state_dict_inflated['transformer.resblocks.' + str(i) + '.attn.in_proj_bias'] = state_dict_2d[
#             'blocks.' + str(i) + '.attn.qkv.bias']
#         state_dict_inflated['transformer.resblocks.' + str(i) + '.attn.out_proj.weight'] = state_dict_2d[
#             'blocks.' + str(i) + '.attn.proj.weight']
#         state_dict_inflated['transformer.resblocks.' + str(i) + '.attn.out_proj.bias'] = state_dict_2d[
#             'blocks.' + str(i) + '.attn.proj.bias']
#         state_dict_inflated['transformer.resblocks.' + str(i) + '.mlp.c_fc.weight'] = state_dict_2d[
#             'blocks.' + str(i) + '.mlp.fc1.weight']
#         state_dict_inflated['transformer.resblocks.' + str(i) + '.mlp.c_fc.bias'] = state_dict_2d[
#             'blocks.' + str(i) + '.mlp.fc1.bias']
#         state_dict_inflated['transformer.resblocks.' + str(i) + '.mlp.c_proj.weight'] = state_dict_2d[
#             'blocks.' + str(i) + '.mlp.fc2.weight']
#         state_dict_inflated['transformer.resblocks.' + str(i) + '.mlp.c_proj.bias'] = state_dict_2d[
#             'blocks.' + str(i) + '.mlp.fc2.bias']
#         for j in range(1,3):
#             state_dict_inflated['transformer.resblocks.' + str(i) + '.ln_'+str(j)+'.weight'] = state_dict_2d[
#                 'blocks.' + str(i) + '.norm'+str(j)+'.weight']
#             state_dict_inflated['transformer.resblocks.' + str(i) + '.ln_'+str(j)+'.bias'] = state_dict_2d[
#                 'blocks.' + str(i) + '.norm'+str(j)+'.bias']
#     return state_dict_inflated

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
        assert attn_type in ['full']
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()


    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

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
        return x

    def forward(self, x, size):
        """[B N C]"""
        x = self._forward_in_frame(x)

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

class TransformerL(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_type='full', enable_checkpoint=False):
        super().__init__()
        self.width = width
        self.layers = layers

        #self.resblocks = MySequential(*[ResidualAttentionBlock(width, heads, attn_type) for _ in range(layers)])
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_type) for _ in range(layers)])
        self.enable_checkpoint = enable_checkpoint
        # self.prompts_add_weights = nn.Parameter(torch.randn(layers-1))
        # nn.init.constant_(self.prompts_add_weights,0.5)
        #nn.init.zeros_(self.prompts_add_weights[:, 1])

    def forward(self, x: torch.Tensor, size,prompts_in,prompts_outs,T,prompts_pos):
        checkpoint_segment = 2
        if self.enable_checkpoint:
            x, size = sequential_checkpoint(self.resblocks, checkpoint_segment, x, size)
            return x
        else:
            if T == 0:
                prompts = torch.cat([prompts_outs, prompts_in], dim=1) + prompts_pos
                x = torch.cat([prompts, x], dim=1)
                for i, blk in enumerate(self.resblocks):
                    x, size = blk(x, size)
            else:
                for i,blk in enumerate(self.resblocks):
                    if i == 0:
                        prompts = torch.cat([prompts_outs[:,:,i],prompts_in], dim=1) + prompts_pos
                        x = torch.cat([prompts, x], dim=1)
                    else:
                        x[:, :4]= 0.5 * prompts_outs[:,:,i]  + 0.5 * x[:,:4]
                        # x_patch = x[:, 4:]
                        # prompt_sum = self.prompts_add_weights[i - 1] * x[:, :4] + (1.0 - self.prompts_add_weights[i - 1]) * prompts_outs[:, :, i]
                        # x = torch.cat([prompt_sum,x_patch], dim=1)
                    x,size = blk(x,size)
            return x

class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, :1]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


# class Block_CA(nn.Module):
#     # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#     # with slight modifications to add CA and LayerScale
#     def __init__(self, dim, num_heads,  qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  norm_layer=nn.LayerNorm, Attention_block=Class_Attention,
#                  ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention_block(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         self.drop_path =  nn.Identity()
#         self.norm2 = norm_layer(dim)
#         #mlp_hidden_dim = int(dim * mlp_ratio)
#         #self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(dim, dim * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(dim * 4, dim))
#         ]))
#         #self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
#         #self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
#
#     def forward(self, x, x_cls):
#         u = torch.cat((x_cls, x), dim=1)
#
#         # x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
#         #
#         # x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
#
#         x_cls = x_cls + self.drop_path(self.attn(self.norm1(u)))
#
#         x_cls = x_cls + self.drop_path(self.mlp(self.norm2(x_cls)))
#
#         return x_cls





class CLIP2DSCLS12PCA(nn.Module):
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
                 use_cls_token=False,
                 ):
        # input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int
        super().__init__()

        # build model
        self.batch_mode = batch_mode
        self.input_resolution = input_resolution
        self.frames = frames
        self.enable_checkpoint = enable_checkpoint
        self.use_text_classifier = use_text_classifier
        self.layers=layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size[1:], stride=patch_size[1:],
                               bias=False)
        scale = width ** -0.5
        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            print("!!!!!use cls token   12pppp!!!!")
            self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size[1]) ** 2 +
                                                                     (1 if self.use_cls_token else 0), width))


        self.prompts_embedding = nn.Parameter(scale * torch.randn(1, 4, width))
        self.prompts_pos = nn.Parameter(scale * torch.randn(1, 4 * 2, width))
        self.prompts_init_out = nn.Parameter(scale * torch.randn(1, 4, width))

        self.prompts_projection = nn.Linear(width, width * layers, bias=False)
        self.prompts_projection_ln = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.prompts_ca=Class_Attention(width, num_heads=heads)
        self.ln_pre = LayerNorm(width)
        self.transformer = TransformerL(width, layers, heads, 'full', enable_checkpoint)
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
        nn.init.normal_(self.prompts_embedding, std=0.02)
        nn.init.normal_(self.prompts_pos, std=0.02)
        nn.init.normal_(self.prompts_init_out, std=0.02)
        if self.temporal_model == 'transformer':
            nn.init.normal_(self.model_t_pos, std=0.02)
        if self.use_cls_token:
            nn.init.normal_(self.class_embedding, std=0.02)


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
       # model_state_dict_2d['positional_embedding'] = model_state_dict_2d['positional_embedding'][1:]

        inflated_model_dict = inflate_weight(model_state_dict_2d, model_state_dict_3d)
        msg = self.load_state_dict(inflated_model_dict, strict=False)
        print(msg)
        print('Pretrained network weights loaded.')
    # def _load_pretrain(self, pretrain):
    #     if pretrain is None or not os.path.exists(pretrain):
    #         return
    #     print(f'Loading network weights from {pretrain}')
    #     model_state_dict_3d = self.state_dict()
    #
    #     clip_model = torch.load(pretrain, map_location='cpu')
    #     #clip_model = clip_model.visual
    #     model_state_dict_2d = clip_model['model']
    #
    #     # remove pos embed for class token
    #    # model_state_dict_2d['positional_embedding'] = model_state_dict_2d['positional_embedding'][1:]
    #
    #     inflated_model_dict = inflate_weight(model_state_dict_2d, model_state_dict_3d)
    #     msg = self.load_state_dict(inflated_model_dict, strict=False)
    #     print(msg)
    #     print('Pretrained network weights loaded.')

    # def _load_pretrain(self, pretrain):
    #     if pretrain is None or not os.path.exists(pretrain):
    #         return
    #     print(f'Loading network weights from {pretrain}')
    #     checkpoint = torch.load(pretrain, map_location='cpu')
    #     load_state_dict = checkpoint['model']
    #     model_state_dict_3d = self.state_dict()
    #
    #     print('inflate weights')
    #     load_state_dict = inflate_weight(load_state_dict, model_state_dict_3d)
    #     msg = self.load_state_dict(load_state_dict, strict=False)
    #     print(f"resume model: {msg}")

    def forward_features(self, x: torch.Tensor):
        B = x.shape[0]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)
        size = list(x.shape[2:])
        T = x.shape[0] // B
        x = rearrange(x, '(b t) c h w -> t b (h w) c', b=B)

        pos_embed = rearrange(self.positional_embedding, 'hw c -> 1 1 hw c')
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], x.shape[1],1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=2)
        x = x + pos_embed

        x = self.ln_pre(x)

        outs = []
        prompts_out_his=[]
        prompts_in = self.prompts_embedding + torch.zeros(B, 1, 1, device=x.device)
        prompts_out = self.prompts_init_out + torch.zeros(B, 1, 1, device=x.device)
        for i in range(T):
            prompts_outs = []
            out = self.transformer(x[i],size,prompts_in,prompts_out,i,self.prompts_pos)
            proj_out = rearrange(self.prompts_projection(out[:, 4:8]), 'b n (layers c) -> b n layers c', layers=self.layers)
            for i,ln in enumerate(self.prompts_projection_ln):
                pout = ln(proj_out[:,:,i])
                prompts_outs.append(pout)
            prompts_out = torch.stack(prompts_outs,dim=2)
            prompts_out_his.append(prompts_out)
            prompts_out_his_array=torch.stack(prompts_out_his,dim=2)
            prompts_out_his_array=rearrange(prompts_out_his_array,'b n t l c -> (b n l) t c')
            prompts_out=self.prompts_ca(prompts_out_his_array)
            prompts_out=rearrange(prompts_out,'(b n l) 1 c -> b n 1 l c',n=4,l=12).squeeze(2)
            if self.use_cls_token:
                outs.append(out[:, 8])
            else:
                outs.append(out[:, 4:8].mean(dim=1))
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