import torch
import torch.nn as nn
from torch.nn import init


def Kaiming_Initialize(model, param):
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m,nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0, 0.01)
            m.bias.data.zero_()


def DeRain_Initialize(net, param):
    """
    Args:
        net: 要初始化的网络
        param: 初始化参数
    在论文EfficientDeRain中，选择默认设置：zero mean Gaussian distribution with a standard deviation of 0.02
    """
    init_type = param["init_type"]  # 初始化方法的名称:normal | Xavier | kaim | orthogonal
    init_gain = param["init_gain"]  # scaling factor for normal, xavier and orthogonal.

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# 权重初始化
def Yolov5_Initialize(model, param):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def ORCRecognize_Initialize(model, param):
    def weights_init(m):
        if isinstance(m, torch.nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            init.constant_(m.bias, 0)
    model.apply(weights_init)
