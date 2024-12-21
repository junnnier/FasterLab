import itertools
import torch.optim as optim
import torch.nn as nn


def SGD(net, param):
    optimizer = optim.SGD(net.parameters(), lr=param["lr"], momentum=param["momentum"], weight_decay=param["weight_decay"])
    return optimizer


def Adam(net, param):
    optimizer = optim.Adam(net.parameters(), lr=param["lr"], betas=(param["momentum1"], param["momentum2"]), weight_decay=param["weight_decay"])  # 调整beta1动量
    return optimizer


def Adan(net, param):
    from extension.Adan import Adan
    optimizer = Adan(net.parameters())
    return optimizer


def CLIPAdamW(net, param):
    params = [
        {"params": net.image_encoder.parameters(), "lr": param["image_encoder_lr"]},
        {"params": net.text_encoder.parameters(), "lr": param["text_encoder_lr"]},
        {"params": itertools.chain(net.image_projection.parameters(), net.text_projection.parameters()),
         "lr": param["head_lr"],
         "weight_decay": param["weight_decay"]}
    ]
    optimizer = optim.AdamW(params, weight_decay=0.0)
    return optimizer


def Yolov5SGD(net, param):
    g0, g1, g2 = [], [], []  # 设定优化器中的参数组
    for v in net.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # 参数有bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # 权重（没有衰减）
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # 权重(衰减)
            g1.append(v.weight)

    optimizer = optim.SGD(g0, lr=param['lr'], momentum=param['momentum'], nesterov=True)
    # 把g1添加到权重衰减中
    optimizer.add_param_group({'params': g1, 'weight_decay': param['weight_decay']})
    optimizer.add_param_group({'params': g2})
    return optimizer


def Yolov5Adam(net, param):
    g0, g1, g2 = [], [], []  # 设定优化器中的参数组
    for v in net.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # 参数有bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # 权重（没有衰减）
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # 权重(衰减)
            g1.append(v.weight)

    optimizer = optim.Adam(g0, lr=param['lr'], betas=(param['momentum'], 0.999))
    # 把g1添加到权重衰减中
    optimizer.add_param_group({'params': g1, 'weight_decay': param['weight_decay']})
    optimizer.add_param_group({'params': g2})
    return optimizer


def DETRAdamW(net, param):
    param_dicts = [
        {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": param["lr_backbone"]
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=param["lr"], weight_decay=param["weight_decay"])
    return optimizer


def DeepLabV3PlusSGD(net, param):
    optimizer = optim.SGD(params=[{'params': net.backbone.parameters(), 'lr': 0.1 * param["lr"]},
                                  {'params': net.classifier.parameters(), 'lr': param["lr"]}],
                          lr=param["lr"],
                          momentum=param["momentum"],
                          weight_decay=param["weight_decay"])
    return optimizer