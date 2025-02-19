from omegaconf import DictConfig

from models.backbone.build_backbone import build_backbone
from models.neck.build_neck import build_neck
from models.head.build_head import build_head

class Model:
    def __init__(self, backbone, neck, head):
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        # 各パーツを連携させる処理の例
        features = self.backbone(x)
        features = self.neck(features)
        output = self.head(features)
        return output

def build_model(cfg: DictConfig):
    
    backbone = build_backbone(cfg.backbone)

    in_channels = backbone.get_stage_dims(cfg.neck.in_stages)
    strides = backbone.get_strides(cfg.neck.in_stages)

    neck = build_neck(cfg.neck, in_channels=in_channels)
    head = build_head(cfg.head, in_channels=in_channels, strides=strides)

    print('inchannels:', in_channels)
    print('strides:', strides)
    
    return Model(backbone, neck, head)
