import torch
from torch import nn
import torch.nn.functional as F
from extension.dbnet_module.backbone import build_backbone
from extension.dbnet_module.neck import build_neck
from extension.dbnet_module.head import build_head


class DBNet(nn.Module):
    def __init__(self, model_config: dict):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        neck_type = model_config.neck.pop('type')
        head_type = model_config.head.pop('type')
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **model_config.neck)
        self.head = build_head(head_type, in_channels=self.neck.out_channels, **model_config.head)
        self.name = f'{backbone_type}_{neck_type}_{head_type}'

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        y = self.head(neck_out)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y


def create_model(param):
    param = {
        'backbone': {'type': 'resnest50', 'pretrained': True, "in_channels": 3},
        'neck': {'type': 'FPN', 'inner_channels': 256},  # 分割头，FPN or FPEM_FFM
        'head': {'type': 'DBHead', 'out_channels': 2, 'k': 50},
    }
    return DBNet(param)


if __name__ == '__main__':
    param = {
        'backbone': {'type': 'resnest50', 'pretrained': True, "in_channels": 3},
        'neck': {'type': 'FPN', 'inner_channels': 256},  # 分割头，FPN or FPEM_FFM
        'head': {'type': 'DBHead', 'out_channels': 2, 'k': 50},
    }
    net = create_model(param)
    result = net(torch.zeros(1, 3, 640, 640))
    print(result.shape)