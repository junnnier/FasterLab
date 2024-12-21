import torch
import torch.nn as nn


class Detect(nn.Module):
    stride = None  # 在构建过程中计算步长
    onnx_dynamic = False  # ONNX模型导出

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # 类别数量
        self.no = nc + 5  # 每个anchor输出的数量
        self.nl = len(anchors)  # 检测层数
        self.na = len(anchors[0]) // 2  # 每一层上anchor的数量
        self.grid = [torch.zeros(1)] * self.nl  # 初始化 grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # 初始化 anchor grid
        # 模型保存参数有两种，一种反向传播被optimizer更新的叫parameter，另一种不需要optimizer更新叫buffer，第二种需要创建tensor，然后将tensor通过register_buffer进行注册
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 最终输出的卷积
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # 推理输出
        for i in range(self.nl):  # 对每一个层i进行操作
            x[i] = self.m[i](x[i])  # 提取预测层的输出送入对应输出的卷积
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # 调整格式顺序，把x(bs,255,20,20)变为x(bs,3,20,20,85)
            # 如果是推理阶段
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)  # 构造网格
                # 计算预测框的坐标信息
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # 调整xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # 调整wh
                z.append(y.view(bs, -1, self.no))  # 预测框的坐标信息

        return x if self.training else (torch.cat(z, 1), x)

    # 划分单元网格
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid