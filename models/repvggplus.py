"""An official improved version of RepVGG in pytorch
[1] Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, Jian Sun.
    RepVGG: Making VGG-style ConvNets Great Again
    https://arxiv.org/abs/2101.03697
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np


def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    result.add_module('relu', nn.ReLU())
    return result


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


# 注意力机制
class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class RepVGGplusStage(nn.Module):
    def __init__(self, in_planes, planes, num_blocks, stride, use_checkpoint, use_post_se=False, deploy=False):
        super(RepVGGplusStage,self).__init__()
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        self.in_planes = in_planes
        for stride in strides:
            cur_groups = 1
            blocks.append(RepVGGplusBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, groups=cur_groups, deploy=deploy, use_post_se=use_post_se))
            self.in_planes = planes
        self.blocks = nn.ModuleList(blocks)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        return x


class RepVGGplusBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_post_se=False):
        super(RepVGGplusBlock,self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        # RepVGG的经典之处只用3*3卷积，padding为1保持特征图大小
        assert kernel_size == 3
        assert padding == 1

        self.nonlinearity = nn.ReLU()

        # 是否添加注意力机制模块
        if use_post_se:
            self.post_se = SEBlock(out_channels, internal_neurons=out_channels // 4)
        else:
            self.post_se = nn.Identity()

        # 是否推理
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=True, padding_mode=padding_mode)
        else:
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.rbr_identity = None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            padding_11 = padding - kernel_size // 2
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, x):
        if self.deploy:
            return self.post_se(self.nonlinearity(self.rbr_reparam(x)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
        out = self.rbr_dense(x) + self.rbr_1x1(x) + id_out
        out = self.post_se(self.nonlinearity(out))
        return out

    # 这个函数以可微分的方式导出等效的核和偏置。可以在任何时候得到等效的内核和偏差
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        # For the 1x1 or 3x3 branch
        if isinstance(branch, nn.Sequential):
            kernel, running_mean, running_var, gamma, beta, eps = branch.conv.weight, branch.bn.running_mean, branch.bn.running_var, branch.bn.weight, branch.bn.bias, branch.bn.eps
        # For the identity branch
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                #   Construct and store the identity kernel in case it is used multiple times
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel, running_mean, running_var, gamma, beta, eps = self.id_tensor, branch.running_mean, branch.running_var, branch.weight, branch.bias, branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepVGGplus(nn.Module):
    def __init__(self, num_blocks, num_classes, width_multiplier, deploy=False, use_post_se=False, use_checkpoint=False, activation=None):
        """
        Args:
            num_blocks: 每个阶段的深度(tuple[int])。
            num_classes: 类别数量
            width_multiplier: 网络中四个阶段的宽度(tuple[float]) (64 * i_0, 128 * i_1, 256 * i_2, 512 * i_3)
            deploy: 默认False,如果为True则模型将进行实时推理结构。
            use_post_se: 默认False,如果为True该模型将在conv-ReLU单元之后进行Squeeze-and-Excitation模块。
            use_checkpoint: 默认False,如果为True模型将在训练期间以可接受的速度保存
        """
        super(RepVGGplus,self).__init__()

        self.deploy = deploy
        self.num_classes = num_classes
        in_channels = min(64, int(64 * width_multiplier[0]))
        stage_channels = [int(64 * width_multiplier[0]), int(128 * width_multiplier[1]), int(256 * width_multiplier[2]), int(512 * width_multiplier[3])]

        self.stage0 = RepVGGplusBlock(in_channels=3, out_channels=in_channels, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_post_se=use_post_se)
        self.stage1 = RepVGGplusStage(in_channels, stage_channels[0], num_blocks[0], stride=2, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        self.stage2 = RepVGGplusStage(stage_channels[0], stage_channels[1], num_blocks[1], stride=2, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        self.stage3_first = RepVGGplusStage(stage_channels[1], stage_channels[2], num_blocks[2] // 2, stride=2, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        self.stage3_second = RepVGGplusStage(stage_channels[2], stage_channels[2], num_blocks[2] - num_blocks[2] // 2, stride=1, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        self.stage4 = RepVGGplusStage(stage_channels[2], stage_channels[3], num_blocks[3], stride=2, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()  # 将连续的几个维度展平成一个tensor（将一些维度合并），这里展开成1维。
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)
        # ----------自定义--------------
        if activation == "softmax":
            self.cls_act = nn.Softmax(dim=-1)
        elif activation == "sigmoid":
            self.cls_act = nn.Sigmoid()
        else:
            self.cls_act = nn.Identity()
        # ------------------------------

        if not self.deploy:
            self.stage1_aux = self._build_aux_for_stage(self.stage1)
            self.stage2_aux = self._build_aux_for_stage(self.stage2)
            self.stage3_first_aux = self._build_aux_for_stage(self.stage3_first)

    def _build_aux_for_stage(self, stage):
        stage_out_channels = list(stage.blocks.children())[-1].rbr_dense.conv.out_channels
        downsample = conv_bn_relu(in_channels=stage_out_channels, out_channels=stage_out_channels, kernel_size=3, stride=2, padding=1)
        fc = nn.Linear(stage_out_channels, self.num_classes, bias=True)
        return nn.Sequential(downsample, nn.AdaptiveAvgPool2d(1), nn.Flatten(), fc)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        stage1_aux = self.stage1_aux(out)
        out = self.stage2(out)
        stage2_aux = self.stage2_aux(out)
        out = self.stage3_first(out)
        stage3_first_aux = self.stage3_first_aux(out)
        out = self.stage3_second(out)
        out = self.stage4(out)
        y = self.gap(out)
        y = self.flatten(y)
        y = self.linear(y)
        # ----------自定义--------------
        y = self.cls_act(y)
        # ------------------------------
        return {'main': y,
                'stage1_aux': stage1_aux,
                'stage2_aux': stage2_aux,
                'stage3_first_aux': stage3_first_aux}

    def switch_repvggplus_to_deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        if hasattr(self, 'stage1_aux'):
            self.__delattr__('stage1_aux')
        if hasattr(self, 'stage2_aux'):
            self.__delattr__('stage2_aux')
        if hasattr(self, 'stage3_first_aux'):
            self.__delattr__('stage3_first_aux')
        self.deploy = True


def create_model(param):
    class_num = param["category"]
    activations = param["activations"]
    return RepVGGplus(num_blocks=[8, 14, 24, 1],
                      num_classes=class_num,
                      width_multiplier=[2.5, 2.5, 2.5, 5],
                      deploy=False,
                      use_post_se=True,
                      use_checkpoint=False,
                      activation=activations)


if __name__ == '__main__':
    from torchsummary import summary
    param = {"category": 2,
             "activations": None}
    net=create_model(param)
    result=summary(net,input_size=(3,256,256))
    print(result)