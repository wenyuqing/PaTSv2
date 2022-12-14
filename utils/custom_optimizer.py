import json
import torch
from typing import List
import torch.optim as optim


T2D_LR_CONFIG = [
    ('_t', 125), ('prompts_', 125), ('head', 125)
]

EVL_LR_CONFIG = [
    ('decoder', 100), ('proj.0', 100),('proj.2', 100), ('prompt', 100), ('visual_proj', 100)
]

def param_groups_paramwise(
        model: torch.nn.Module,
        paramwise_cfg: List,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        no_weight_decay_list: set = {},
        zero_wd_1d_param: bool = True
):
    param_group_names = {}
    param_groups = {}
    paramwise_cfg = {k: v for k, v in paramwise_cfg}
    # first sort with alphabet order and then sort with reversed len of str
    # sorted_keys = sorted(sorted(paramwise_cfg.keys()), key=len, reverse=True)
    # NOTE: we switch to the input order, the front item will be matched first.
    sorted_keys = paramwise_cfg.keys()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (zero_wd_1d_param and (param.ndim == 1 or name.endswith(".bias"))) or \
            name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        this_scale = 1.0
        for key in sorted_keys:
            if key in name:
                this_scale = paramwise_cfg[key]
                break
        group_name = "scale_%.5f_%s" % (this_scale, g_decay)
        if group_name not in param_group_names:
            param_group_names[group_name] = {
                "lr": lr * this_scale,
                "weight_decay": this_decay,
                "param_names": [],
            }
            param_groups[group_name] = {
                "lr": lr * this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
        param_group_names[group_name]["param_names"].append(name)
        param_groups[group_name]["params"].append(param)

    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    return list(param_groups.values())


def build_optimizer_t2d(config, model, paramwise_cfg):
    model = model.module if hasattr(model, 'module') else model

    skip_keywords = {}
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameter_groups = param_groups_paramwise(model, paramwise_cfg, config.TRAIN.LR, config.TRAIN.WEIGHT_DECAY, skip_keywords)

    optimizer = optim.AdamW(parameter_groups, betas=(0.9, 0.98), eps=1e-8, )

    return optimizer