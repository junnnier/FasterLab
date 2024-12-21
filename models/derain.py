import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                                   # nn.BatchNorm2d(out_ch),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                                   # nn.BatchNorm2d(out_ch),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                                   # nn.BatchNorm2d(out_ch),
                                   nn.ReLU()
                                   )
        if channel_att:
            self.att_c = nn.Sequential(nn.Conv2d(2*out_ch, out_ch//g, 1, 1, 0),
                                       nn.ReLU(),
                                       nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                                       nn.Sigmoid()
                                       )
        if spatial_att:
            self.att_s = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                                       nn.Sigmoid()
                                       )

    def forward(self, data):
        fm = self.conv1(data)
        if self.channel_att:
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm


class KernelConv(nn.Module):
    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)  # 从小到大排序
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core, 2p --> p^2
        Args:
            core: batch*(N*2*K)*height*width
        Returns:
        """
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, cur:cur + K, ...].view(batch_size, N, K, 1, 3, height, width)
            t2 = core_2[:, :, cur:cur + K, ...].view(batch_size, N, 1, K, 3, height, width)
            core_out[K] = torch.einsum('ijklno,ijlmno->ijkmno', [t1, t2]).view(batch_size, N, K * K, color, height, width)
            cur += K
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        Args:
            core: [batch_size, (burst_length*K*K), height, width]
        Returns: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size[0]] = core[:, :, 0:self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0, rate=1):
        """
        compute the pred image according to core and frames
        Args:
            frames: [batch_size, 3, height, width]
            core: [batch_size, K*K*burst_length, height, width]
        Returns:
        """
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)  # 类似于把图片按3个通道拆分了[batch_size, 3, 1, height, width]
        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)  # [batch_size, 3, 9, 1, height, width]
        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]  # 从大的卷积核开始取
        # 对每个设定的卷积核操作
        for index, K in enumerate(kernel):
            if not img_stack:
                padding_num = (K//2) * rate  # 计算当前卷积核下，当前rate扩展填充图像边缘的大小，rate越大，需要填充的越多
                frame_pad = F.pad(frames, [padding_num, padding_num, padding_num, padding_num])
                # 根据填充后的图，从上到下从左到右，每次移动1格，裁剪成k*k张图
                for i in range(0, K):
                    for j in range(0, K):
                        img_stack.append(frame_pad[..., i*rate:i*rate + height, j*rate:j*rate + width])
                img_stack = torch.stack(img_stack, dim=2)  # 堆叠后[batch_size, 3, 1, height, width] -> [batch_size, 3, 9, 1, height, width]
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]
            pred_img.append(torch.sum(core[K].mul(img_stack), dim=2, keepdim=False))  # [batch_size, 3, 1, height, width]
        pred_img = torch.stack(pred_img, dim=0)
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)
        pred_img_i = pred_img_i.squeeze(2)

        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i += bias

        pred_img_i = pred_img_i / white_level
        # pred_img = torch.mean(pred_img_i, dim=1, keepdim=True)

        return pred_img_i


class KPN(nn.Module):
    def __init__(self, color=True, burst_length=1, blind_est=True, kernel_size=[3], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        """
        Args:
            color: 输入网络的是否为彩色图片
            burst_length: 在连拍设置中使用的照片数量
            blind_est: 差异图
            kernel_size: 卷积核大小
            sep_conv: 简单的输出类型
            channel_att: 通道注意力
            spatial_att: 空间注意力
            upMode: 上采样模式
            core_bias: 核心偏置
        """
        super(KPN, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = self.color_channel * (burst_length if blind_est else burst_length + 1)
        out_channel = self.color_channel * (2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length
        if core_bias:
            out_channel += self.color_channel * burst_length
        # 2~5层都是均值池化+3层卷积
        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.conv6 = Basic(512 + 512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256 + 512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256 + 128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)

        self.conv_final = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

    # 前向传播函数
    def forward(self, data_with_est, white_level=1.0):
        """
        Args:
            data_with_est: if not blind estimation, it is same as data
            data: input image (临时去除这个输入参数，保证只有一个数据输入网络)
        Returns: predict result
        """
        data = data_with_est.clone().detach()
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # 开始上采样  同时要进行skip connection
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode, align_corners=False)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.upMode, align_corners=False)], dim=1))
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode, align_corners=False)], dim=1))
        # return channel K*K*N
        core = self.outc(F.interpolate(conv8, scale_factor=2, mode=self.upMode, align_corners=False))

        pred1 = self.kernel_pred(data, core, white_level, rate=1)
        pred2 = self.kernel_pred(data, core, white_level, rate=2)
        pred3 = self.kernel_pred(data, core, white_level, rate=3)
        pred4 = self.kernel_pred(data, core, white_level, rate=4)

        pred_cat = torch.cat([torch.cat([torch.cat([pred1, pred2], dim=1), pred3], dim=1), pred4], dim=1)

        pred = self.conv_final(pred_cat)

        return pred


def create_model(param):
    return KPN(color=True, burst_length=param["burst_length"], blind_est=True, kernel_size=param["kernel_size"], sep_conv=False,
               channel_att=False, spatial_att=False, upMode=param["upMode"], core_bias=False)


if __name__ == '__main__':
    from torchsummary import summary
    param = {}
    net=create_model(param)
    result=summary(net,input_size=(3,299,299))
    print(result)