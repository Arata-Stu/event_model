from typing import Tuple

import torch 
from omegaconf import DictConfig

from .yolo_pafpn import YOLOPAFPN

def build_neck(neck_cfg: DictConfig, in_channels: Tuple[int, ...]) -> torch.nn.Module:
    if neck_cfg.name == 'PAFPN':
        neck = YOLOPAFPN(in_channels=in_channels,
                         depthwise=neck_cfg.depthwise,
                         act=neck_cfg.act,
                         inplanes=neck_cfg.inplanes)
    else:
        raise NotImplementedError(f"Neck {neck_cfg.name} is not implemented")
    return neck
