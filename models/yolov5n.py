import torch
import torch.nn as nn
import math
import thop
# --------本地导入--------
from extension.yolov5_module.common import (Conv, Concat, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv,
                                            Focus, BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, Contract, Expand)
from extension.yolov5_module.head import Detect
from extension.yolov5_module.tools import model_info, fuse_conv_and_bn, feature_visualization, make_divisible
from utils.torch_tools import time_sync


class YoLoV5(nn.Module):
    def __init__(self, structure):
        super().__init__()
        # 根据配置解析模型，模型推理中保存的数据列表
        self.model, self.save = self.parse_model(structure, ch=[3])
        # 默认类别名
        self.names = {i: f'{i}' for i in range(structure["nc"])}
        self.inplace = structure["inplace"]
        # 设定Detect层的stride(即步长，类似下采样的倍率)
        m = self.model[-1]
        s = 256  # 2x min stride
        m.inplace = self.inplace
        m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, 3, s, s))])  # 前向推理一次，计算得到步长
        m.anchors /= m.stride.view(-1, 1, 1)  # 根据步长调整对应锚框的值
        self.stride = m.stride
        self._initialize_biases()

    # 解析模型
    def parse_model(self, d, ch):
        """
        Args:
            d: 模型结构字典
            ch: 模型输入通道数
        Returns:
        """
        # 打印信息的表头
        print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments(ch_input, ch_output, kernel(C3:number), stride, padding)'))
        anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # 单个网格里anchors的数量
        no = na * (nc + 5)  # 最后outputs通道的数量 = anchors * (classes + 5)
        layers, save, c2 = [], [], ch[-1]  # 层名，保存特征索引，上一层的通道数

        # 对网络模块循环设置 from, number, module, args
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
            m = eval(m) if isinstance(m, str) else m  # 对参数module进行eval(strings)
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a  # 对参数args里的内容进行eval(strings)
                except NameError:
                    pass
            n = n_ = max(round(n * gd), 1) if n > 1 else n  # 模块深度增益计算，即需要多少个模块
            # 根据配置，对所属的每个模块进行设置
            if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus, BottleneckCSP, C3, C3TR,
                     C3SPP, C3Ghost]:
                c1, c2 = ch[f], args[0]  # 获取来自f层的输出通道数作为输入，和该层需要输出的通道数，f=-1就是上一层
                if c2 != no:  # 如果不是最后输出的通道数
                    c2 = make_divisible(c2 * gw, 8)  # 进行分割，修改模块宽度，即输出特征图的通道数

                args = [c1, c2, *args[1:]]  # 更新args参数
                if m in [BottleneckCSP, C3, C3TR, C3Ghost]:  # 如果是这几个模块，增加模块串联的数量
                    args.insert(2, n)
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:  # 拼接操作
                c2 = sum([ch[x] for x in f])  # 通道累计
            elif m is Detect:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
            elif m is Contract:
                c2 = ch[f] * args[0] ** 2
            elif m is Expand:
                c2 = ch[f] // args[0] ** 2
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # 创建模块
            t = str(m)[8:-2].replace('__main__.', '')  # 修改模块名字
            np = sum([x.numel() for x in m_.parameters()])  # 统计模块参数，numle()获取张量元素个数
            m_.i, m_.f, m_.type, m_.np = i, f, t, np  # 存入模块名等 attach index, 'from' index, type, number params
            print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, np, t, args))  # 打印信息
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 添加到savelist
            layers.append(m_)  # 模块保存到layers列表
            if i == 0:
                ch = []
            ch.append(c2)  # 把每一层的输出的通道数加入到ch
        return nn.Sequential(*layers), sorted(save)

    def forward(self, x, augment=False, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)

    # 单次前向推理
    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # 如果不是来自上一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 从更前面层的输出保存到y中找
            if profile:  # 是否进行性能评估flops
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # 如果这一模块在要求的save中，保存输出到y，用于后面的使用
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        """
        在给定的输入上描述模型的单个层的计算时间和FLOPs。将结果追加到所提供的列表。
        Args:
            m (nn.Module): 模块。
            x (torch.Tensor): 输入模块的tensor。
            dt (list): 存储层计算时间的列表。
        """
        c = isinstance(m, Detect)
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            print(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        print(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            print(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    # 融合Conv2d()层和BatchNorm2d()层
    def fuse(self):
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 进行融合
                delattr(m, 'bn')  # 移除batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    # 打印模型信息
    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)

    def _initialize_biases(self, cf=None):
        #  初始化Detect()的偏置，Cf是类别频率 https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def yolov5_base(class_num, depth_multiple, width_multiple, anchors):
    structure = {
        "inplace": True,
        "nc": class_num,
        "depth_multiple": depth_multiple,  # module depth multiple
        "width_multiple": width_multiple,  # layer channel multiple
        "anchors": anchors,
        "backbone": [
            # [from, number, module, args]
            [-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
            [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
            [-1, 3, C3, [128]],
            [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
            [-1, 6, C3, [256]],
            [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
            [-1, 9, C3, [512]],
            [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
            [-1, 3, C3, [1024]],
            [-1, 1, SPPF, [1024, 5]]  # 9
        ],
        "head": [
            [-1, 1, Conv, [512, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 6], 1, Concat, [1]],  # cat backbone P4
            [-1, 3, C3, [512, False]],  # 13
            [-1, 1, Conv, [256, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 4], 1, Concat, [1]],  # cat backbone P3
            [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
            [-1, 1, Conv, [256, 3, 2]],
            [[-1, 14], 1, Concat, [1]],  # cat head P4
            [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
            [-1, 1, Conv, [512, 3, 2]],
            [[-1, 10], 1, Concat, [1]],  # cat head P5
            [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
            [[17, 20, 23], 1, Detect, [class_num, anchors]]  # Detect(P3, P4, P5)
        ]
    }

    return YoLoV5(structure)


def create_model(param):
    class_num = param["category"]
    depth_multiple = 0.33
    width_multiple = 0.25
    anchors = [[10, 13, 16, 30, 33, 23],  # P3/8
               [30, 61, 62, 45, 59, 119],  # P4/16
               [116, 90, 156, 198, 373, 326]]  # P5/32
    return yolov5_base(class_num, depth_multiple, width_multiple, anchors)


if __name__ == '__main__':
    param = {"category": 2}
    net = create_model(param)
    result = net(torch.zeros(1, 3, 512, 512))
    for index, feature in enumerate(result):
        print("{}--{}".format(index, feature.shape))