from typing import Tuple

import torch 
from omegaconf import DictConfig

from .yolox_darknet import CSPDarknet

def build_backbone(backbone_cfg: DictConfig) -> torch.nn.Module:
    if backbone_cfg.name == 'MaxViTRNN':
        backbone = None
        pass
    elif backbone_cfg.name == 'CSPDarknet':
        backbone = CSPDarknet(depth=backbone_cfg.depth,
                              width=backbone_cfg.width,
                              input_dim=backbone_cfg.input_dim,
                              out_features=backbone_cfg.out_features,
                              depthwise=backbone_cfg.depthwise,
                              act=backbone_cfg.act,
                              in_res_hw=backbone_cfg.in_res_hw)
    else:
        raise NotImplementedError(f"Backbone {backbone.name} is not implemented")
    
    return backbone
