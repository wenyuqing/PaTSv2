#!/usr/bin/env python

from typing import Dict, Iterable, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_transformer import QuickGELU, Attention
from .weight_loaders import weight_loader_fn_dict
from .vision_transformer import PatchEmbed2D,LayerNorm,TransformerDecoderLayer,TransformerEncoderLayer
import os
import json
from einops import rearrange
from collections import defaultdict


def inflate_weight(state_dict_2d, state_dict_3d):
    # copy from slowfast.checkpoint
    from collections import OrderedDict
    state_dict_inflated = OrderedDict()
    for k, v2d in state_dict_2d.items():
        if k not in state_dict_3d.keys():
            print(f"Unknown key {k} from 2d dict")
            continue
        v3d = state_dict_3d[k]
        if k=='pos_embed':
            v3d[1:] = v2d
        # Inflate the weight of 2D conv to 3D conv.
        # if len(v2d.shape) == 4 and len(v3d.shape) == 5:
        #     print(
        #         "Inflate {}: {} -> {}: {}".format(k, v2d.shape, k, v3d.shape)
        #     )
        #     # Dimension need to be match.
        #     assert v2d.shape[-2:] == v3d.shape[-2:]
        #     assert v2d.shape[:2] == v3d.shape[:2]
        #     v3d = (
        #             v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
        #     )
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
class TemporalCrossAttentionPrompt(nn.Module):

    def __init__(
            self,
            spatial_size: Tuple[int, int] = (14, 14),
            feature_dim: int = 768,
    ):
        super().__init__()

        self.spatial_size = spatial_size

        w_size = np.prod([x * 2 - 1 for x in spatial_size])
        self.w1 = nn.Parameter(torch.zeros([w_size, feature_dim]))
        self.w2 = nn.Parameter(torch.zeros([w_size, feature_dim]))

        idx_tensor = torch.zeros([np.prod(spatial_size) for _ in (0, 1)], dtype=torch.long)
        for q in range(np.prod(spatial_size)):
            qi, qj = q // spatial_size[1], q % spatial_size[1]
            for k in range(np.prod(spatial_size)):
                ki, kj = k // spatial_size[1], k % spatial_size[1]
                i_offs = qi - ki + spatial_size[0] - 1
                j_offs = qj - kj + spatial_size[1] - 1
                idx_tensor[q, k] = i_offs * (spatial_size[1] * 2 - 1) + j_offs
        self.idx_tensor = idx_tensor

    def forward_half(self, q: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        q, k = q[:, :, 1:], k[:, :, 1:]  # remove cls token and prompt token

        assert q.size() == k.size()
        assert q.size(2) == np.prod(self.spatial_size)

        attn = torch.einsum('ntqhd,ntkhd->ntqkh', q / (q.size(-1) ** 0.5), k)
        attn = attn.softmax(dim=-2).mean(dim=-1)  # L, L, N, T

        self.idx_tensor = self.idx_tensor.to(w.device)
        w_unroll = w[self.idx_tensor]  # L, L, C
        ret = torch.einsum('ntqk,qkc->ntqc', attn, w_unroll)

        return ret

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        N, T, L, H, D = q.size()
        # print(q.size(), "...",self.spatial_size)
        assert L == np.prod(self.spatial_size) + 1

        ret = torch.zeros([N, T, L, self.w1.size(-1)], device='cuda')
        ret[:, 1:, 1:, :] = ret[:, 1:, 1:, :] + self.forward_half(q[:, 1:, :, :, :], k[:, :-1, :, :, :], self.w1)
        ret[:, :-1, 1:, :] = ret[:, :-1, 1:, :] + self.forward_half(q[:, :-1, :, :, :], k[:, 1:, :, :, :], self.w2)

        return ret

class EVLDecoderPrompt(nn.Module):

    def __init__(
            self,
            num_frames: int = 8,
            spatial_size: Tuple[int, int] = (14, 14),
            num_layers: int = 4,
            in_feature_dim: int = 768,
            qkv_dim: int = 768,
            num_heads: int = 12,
            mlp_factor: float = 4.0,
            enable_temporal_conv: bool = True,
            enable_temporal_pos_embed: bool = True,
            enable_temporal_cross_attention: bool = True,
            mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.enable_temporal_conv = enable_temporal_conv
        self.enable_temporal_pos_embed = enable_temporal_pos_embed
        self.enable_temporal_cross_attention = enable_temporal_cross_attention
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(in_feature_dim, qkv_dim, num_heads, mlp_factor, mlp_dropout) for _ in
             range(num_layers)]
        )

        if enable_temporal_conv:
            self.temporal_conv = nn.ModuleList(
                [nn.Conv1d(in_feature_dim, in_feature_dim, kernel_size=3, stride=1, padding=1, groups=in_feature_dim)
                 for _ in range(num_layers)]
            )
        if enable_temporal_pos_embed:
            self.temporal_pos_embed = nn.ParameterList(
                [nn.Parameter(torch.zeros([num_frames, in_feature_dim])) for _ in range(num_layers)]
            )
        if enable_temporal_cross_attention:
            self.cross_attention = nn.ModuleList(
                [TemporalCrossAttentionPrompt(spatial_size, in_feature_dim) for _ in range(num_layers)]
            )

        self.cls_token = nn.Parameter(torch.zeros([in_feature_dim]))

    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, in_features: List[Dict[str, torch.Tensor]]):
        N, T, L, C = in_features[0]['out'].size()
        assert len(in_features) == self.num_layers
        x = self.cls_token.view(1, 1, -1).repeat(N, 1, 1)

        for i in range(self.num_layers):
            frame_features = in_features[i]['out']

            if self.enable_temporal_conv:
                feat = in_features[i]['out']
                feat = feat.permute(0, 2, 3, 1).contiguous().flatten(0, 1)  # N * L, C, T
                feat = self.temporal_conv[i](feat)
                feat = feat.view(N, L, C, T).permute(0, 3, 1, 2).contiguous()  # N, T, L, C
                frame_features = frame_features + feat

            if self.enable_temporal_pos_embed:
                frame_features = frame_features + self.temporal_pos_embed[i].view(1, T, 1, C)

            if self.enable_temporal_cross_attention:
                frame_features = frame_features + self.cross_attention[i](in_features[i]['q'], in_features[i]['k'])

            frame_features = frame_features.flatten(1, 2)  # N, T * L, C

            x = self.decoder_layers[i](x, frame_features)

        return x

class EVLTransformer2STREAM(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        backbone_name: str = 'ViT-B/16',
        backbone_type: str = 'clip',
        backbone_path: str = '',
        backbone_mode: str = 'finetune',
        decoder_num_layers: int = 4,
        decoder_qkv_dim: int = 768,
        decoder_num_heads: int = 12,
        decoder_mlp_factor: float = 4.0,
        num_classes: int = 400,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = True,
        cls_dropout: float = 0.5,
        decoder_mlp_dropout: float = 0.5,
        ##backbone cfg
        return_all_features: bool = True,
        act: nn.Module = QuickGELU,
        feature_dim: int = 768,
        input_size : Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_factor: float = 4.0,
        ln_pre: bool = True,
    ):
        super().__init__()

        self.decoder_num_layers = decoder_num_layers

        #backbone_config = self._create_backbone(backbone_name, backbone_type, backbone_path, backbone_mode)

        ###backbone
        self.backbone_type = backbone_type
        self.backbone_path = backbone_path
        self.return_all_features = return_all_features

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, embed_dim=feature_dim)
        self.num_patches = np.prod([x // y for x, y in zip(input_size, patch_size)]) + 2

        self.cls_token = nn.Parameter(torch.zeros([feature_dim]))
        self.prompt = nn.Parameter(torch.zeros([feature_dim]))
        self.pos_embed = nn.Parameter(torch.zeros([self.num_patches, feature_dim]))

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                in_feature_dim=feature_dim, qkv_dim=feature_dim, num_heads=num_heads, mlp_factor=mlp_factor, act=act,
                return_all_features=return_all_features,
            ) for _ in range(num_layers)
        ])

        if ln_pre:
            self.ln_pre = LayerNorm(feature_dim)
        else:
            self.ln_pre = nn.Identity()
        assert backbone_mode == 'finetune'
        assert backbone_mode in ['finetune', 'freeze_fp16', 'freeze_fp32']
        self.freeze = backbone_mode
        if self.freeze == 'freeze_fp16':
            self.model_to_fp16()

        ###backbone

        backbone_feature_dim = feature_dim
        backbone_spatial_size = tuple(x // y for x, y in zip(input_size, patch_size))

        self.decoder = EVLDecoderPrompt(
            num_frames=num_frames,
            spatial_size=backbone_spatial_size,
            num_layers=decoder_num_layers,
            in_feature_dim=backbone_feature_dim,
            qkv_dim=decoder_qkv_dim,
            num_heads=decoder_num_heads,
            mlp_factor=decoder_mlp_factor,
            enable_temporal_conv=enable_temporal_conv,
            enable_temporal_pos_embed=enable_temporal_pos_embed,
            enable_temporal_cross_attention=enable_temporal_cross_attention,
            mlp_dropout=decoder_mlp_dropout,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(backbone_feature_dim),
            nn.Dropout(cls_dropout),
            nn.Linear(backbone_feature_dim, num_classes),
        )

        self._initialize_weights()
        self._load_pretrain(backbone_path)
        self.freeze_clip(self.freeze)

    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.prompt, std=0.02)

    def model_to_fp16(self):
        def _module_to_fp16(m: nn.Module):
            if isinstance(m, (nn.Linear,)):
                m.half()

        self.apply(_module_to_fp16)

        self.pos_embed.data = self.pos_embed.data.half()
        self.cls_token.data = self.cls_token.data.half()

    def freeze_clip(self, freeze):
        if freeze == 'finetune':
            return
        NON_FREEZABLE = ['decoder','proj.0','proj.2']
        frozen_param = []
        trainable_param = []
        for n, p in self.named_parameters():
            #print("all named parameters",n)
            _freeze = True
            for x in NON_FREEZABLE:
                if x in n:
                    _freeze = False
                    break
            if _freeze:
                p.requires_grad = False
                frozen_param.append(n)
            else:
                trainable_param.append(n)
        print('frozen params:')
        print(json.dumps(frozen_param, indent=2))
        print('trainable params:')
        print(json.dumps(trainable_param, indent=2))

    def _load_pretrain(self, pretrain):
        if pretrain is None or not os.path.exists(pretrain):
            return
        print(f'Loading network weights from {pretrain}')
        weight_loader_fn = weight_loader_fn_dict[self.backbone_type]
        model_state_dict_2d = weight_loader_fn(self.backbone_path)
        model_state_dict_3d = self.state_dict()
        #
        # clip_model = torch.jit.load(pretrain, map_location='cpu')
        # clip_model = clip_model.visual
        # model_state_dict_2d = clip_model.state_dict()
        #
        inflated_model_dict = inflate_weight(model_state_dict_2d, model_state_dict_3d)
        msg = self.load_state_dict(inflated_model_dict, strict=False)
        print(msg)
        print('Pretrained network weights loaded.')


    def forward_feature(self, x: torch.Tensor,prompt):
        x = self.patch_embed(x)
        x = torch.cat([prompt, self.cls_token.view(1, 1, -1).repeat(x.size(0), 1, 1), x], dim=1)
        x = x + self.pos_embed

        x = self.ln_pre(x)

        if self.return_all_features:
            all_features = []
            for blk in self.blocks:
                x = blk(x)
                all_features.append(x)
                x = x['out']
            return all_features

        else:
            for blk in self.blocks:
                x = blk(x)
            return x

    def forward(self, x, text):
        #backbone = self._get_backbone(x)
        # b t c h w -> b c t h w

        B, T, C, H, W = x.size()
        xs = torch.unbind(x, dim=1)
        features_bank = [defaultdict(list) for _ in range(self.decoder_num_layers)]

        prompt = self.prompt.view(1, 1, -1).repeat(B, 1, 1)
        for x in xs:
            features = self.forward_feature(x, prompt)
            prompt = features[-1]['out'][:, 0:1, :]
            for l, feature_dict in enumerate(features[-self.decoder_num_layers:]):
                for k, v in feature_dict.items():
                    features_bank[l][k].append(v[:, 1:])
        for l in range(self.decoder_num_layers):
            for k in features_bank[l].keys():
                features_bank[l][k] = torch.stack(features_bank[l][k], dim=1)
        # for t in range(T):
        #     frame_t = x[:,:,t,:,:].contiguous()
        #     if t==0:
        #         features = self.forward_feature(frame_t,t,None)[-self.decoder_num_layers:]
        #         prompt_bank = features[-1]['out'][:, 0, :].unsqueeze(1)
        #     else:
        #         features = self.forward_feature(frame_t,t,prompt_bank)[-self.decoder_num_layers:]
        #         prompt_bank = features[-1]['out'][:, :,0, :].unsqueeze(1)
        #
        #     if t==0:
        #         features_bank=features
        #         for layer in features_bank:
        #             layer['q'] = layer['q'].unsqueeze(1)
        #             layer['k'] = layer['k'].unsqueeze(1)
        #             layer['v'] = layer['v'].unsqueeze(1)
        #             layer['attn_out'] = layer['attn_out'].unsqueeze(1)
        #             layer['out'] = layer['out'].unsqueeze(1)
        #     else:
        #         for i,layer in enumerate(features_bank):
        #             layer['q']=torch.cat((layer['q'],features[i]['q'].unsqueeze(1)),dim=1)
        #             layer['k'] = torch.cat((layer['k'], features[i]['k'].unsqueeze(1)), dim=1)
        #             layer['v'] = torch.cat((layer['v'], features[i]['v'].unsqueeze(1)), dim=1)
        #             layer['attn_out'] = torch.cat((layer['attn_out'], features[i]['attn_out'].unsqueeze(1)), dim=1)
        #             layer['out'] = torch.cat((layer['out'], features[i]['out'].unsqueeze(1)), dim=1)
        # features_bank = [
        #     dict((k, v.float()) for k, v in x.items())
        #     for x in features_bank
        # ]
        x = self.decoder(features_bank)
        x = self.proj(x[:, 0, :])

        return x