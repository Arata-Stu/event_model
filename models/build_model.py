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

def build_model(cfg):
    """
    cfgは以下のような構成を想定：
    {
        'backbone': { 'type': 'resnet', ... },
        'neck': { 'type': 'fpn', ... },
        'head': { 'type': 'detection', ... }
    }
    """
    backbone = build_backbone(cfg['backbone'])
    neck = build_neck(cfg['neck'])
    head = build_head(cfg['head'])
    
    return Model(backbone, neck, head)
