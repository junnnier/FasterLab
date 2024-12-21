import torch
import torch.nn as nn
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
from thop import profile  # 估算pytorch模型的flops模块
import numpy as np
import time
import warnings
import torchvision
# --------本地导入--------
from utils.visual import draw_pr_curve, draw_mc_curve
from utils.label_tools import xywh2xyxy


def model_info(model, verbose=False, img_size=640):
    """
    打印模型信息
    Args:
        model: 网络模型
        verbose: 显示更多冗余信息
        img_size: 图片shape，可以是int或list，即Img_size=640或Img_size=[640, 320]
    """
    n_p = sum(x.numel() for x in model.parameters())  # 参数量
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # 要计算梯度的参数量
    # 是否显示冗长信息
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    # 计算FLOPs
    try:
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPs
    except (ImportError, Exception):
        fs = ''
    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def fuse_conv_and_bn(conv, bn):
    """
    融合conv层和BN层加快运算速度，参考https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    Args:
        conv:
        bn:
    Returns:
    """
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)
    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def feature_visualization(x, module_type, stage, n=32, save_dir="runs/visual_feature"):
    """
    可视化给定模型模块的特征映射。
    Args:
        x (torch.Tensor): 要可视化的特征。
        module_type (str): 模块类型。
        stage (int): 模块在模型中的索引（即第几个阶段）。
        n (int, optional): 要绘制的特征映射的最大数量，默认为32。
        save_dir (Path, optional): 保存结果目录，默认"runs/visual_feature"。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if "Detect" in module_type:
        return
    batch, channels, height, width = x.shape
    if height > 1 and width > 1:
        f = os.path.join(save_dir, "stage{}_{}_features.png".format(stage, module_type.split('.')[-1]))  # 保存图片名

        blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
        n = min(n, channels)  # 绘制数量
        fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 在画布上的子图分布 8行 x n/8列
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(n):
            ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
            ax[i].axis('off')

        plt.savefig(f, dpi=300, bbox_inches='tight')
        plt.close()
        # np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())
        print("Saving {}... ({}/{})".format(f, n, channels))


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


# 根据pr曲线计算ap
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # 计算精度包络线，把曲线填补鼓起来，例如numpy.array([11,12,13,20,19,18,23,21])，输出numpy.array([11,12,13,20,20,20,23,23])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 曲线下积分面积
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point插值(COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # trapz()进行积分
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # 获取recall区间上变化的点
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve，计算曲线下面积

    return ap, mpre, mrec


def accuracy_fitness(x):
    # 将指标的加权组合作为模型的最佳适应精度
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    x = x if isinstance(x, np.ndarray) else np.array(x)
    return (x.reshape((1, -1)) * w).sum(1).item()


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算box1[1,4]和box2[n,4]的IOU值
    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in (x1, y1, x2, y2) format.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # 获取box坐标
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 交叉区域面积
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # 合并的面积
    union = w1 * h1 + w2 * h2 - inter + eps
    # iou值
    iou = inter / union
    # 需要使用哪一种iou
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # # 最小包围框的w
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # 最小包围框的h
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # 包围框的斜边平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # 两个box中心距离的平方
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # 最小包围框的面积
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


class ConfusionMatrix(object):
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # 类别数量
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(os.path.join(save_dir,'confusion_matrix.png'), dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# 非极大值抑制(NMS)
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """
    Returns:
         检测列表, shape为(n,6)的tensor，每张图为[xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # 类别数量
    xc = prediction[..., 4] > conf_thres  # 大于置信度阈值框的索引，shape为[batch, N]

    # 检查设置的置信度阈值是否有效
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # 初始设置
    min_wh, max_wh = 2, 4096  # (像素)最小和最大框的宽和高
    max_nms = 30000  # 进入nms最大检测box的数量 torchvision.ops.nms()
    time_limit = 10.0  # 超过多少秒后退出
    redundant = True  # 是否需要冗余的检测
    multi_label &= nc > 1  # 每个box是否可以有多个标签(增加0.5ms/img)
    merge = False  # 是否使用merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]  # 用于存放每张图的结果
    # 对batch中每张图的预测结果操作
    for xi, x in enumerate(prediction):
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # 对box的width-height限制
        x = x[xc[xi]]  # 根据索引，提取第xi张图中预测结果大于置信度阈值的锚点

        # 如果没有，处理下一个图像
        if not x.shape[0]:
            continue

        # 计算类别框的conf
        x[:, 5:] *= x[:, 4:5]  # conf = 是否有目标的conf * 类别的conf

        # Box从(center x, center y, width, height)调整到(x1, y1, x2, y2)，在输入图片上的位置。
        box = xywh2xyxy(x[:, :4])

        # 一个box是否可以有多个类别
        if multi_label:
            # 类别分数大于阈值就作为一个box
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)  # shape为nx6 (xyxy, conf, cls)
        else:
            # 只取类别分数最大的那个作为该box的类别
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # 指定过滤哪个类别
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # box的数量
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # 按置信度排序去除超出的box

        # NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 乘上一个很大数，对类别离散化
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # 进行NMS，返回过滤后的box索引，降序排列
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        # 是否超出单图最大框数
        if i.shape[0] > max_det:
            i = i[:max_det]
        # 是否使用Merge-NMS: boxes合并使用加权平均值
        if merge and (1 < n < 3E3):
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou矩阵
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        # 提取nms过滤后的box
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output


def process_batch(detections, labels, iouv):
    """
    返回正确的预测矩阵。两组box都是(x1, y1, x2, y2) 格式.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    # 在各iou值下是否预测正确的矩阵[N, 10]
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])  # 计算iou值,shape为[N, M]
    # IoU高于0.5~0.95阈值且类别匹配的坐标索引
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    # 如果索引坐标存在，记录
    if x[0].shape[0]:
        # 调整索引格式变成匹配矩阵，[detection, label, iou]
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        # 如果有多个进行从小到大排序，并去重
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


# 计算每个类别的ap，根据查全率R和查准率P曲线，计算平均查准率。
def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ 计算平均精度，给定召回率和精度曲线。来源: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray). 置信度
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # 对objectness排序，从小到达排序，加符号后conf大的索引就在前面
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 有多少种类别
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # 类别数量

    # 创建Precision-Recall曲线并计算每个类的AP
    px, py = np.linspace(0, 1, 1000), []  # 用于绘图
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))  # 初始化操作,ap的shape是（类别数,10）其中10是mAP0.5~0.95一个10个
    # 对每个类别计算ap
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c  # 在预测类别中找到是当前这个类的索引，值为ture
        n_l = (target_cls == c).sum()  # 计算这个类ground truth有多少个
        n_p = i.sum()  # 这个类预测出来的总个数

        if n_p == 0 or n_l == 0:
            continue
        else:
            # 累计 FP 和 TP
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # 计算 Recall
            recall = tpc / (n_l + 1e-16)  # recall 曲线
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # interp()是线性插值, negative x, xp because xp decreases

            # 计算 Precision
            precision = tpc / (tpc + fpc)  # precision 曲线
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # 通过recall-precision曲线计算AP，当前类的ap0.5~0.95
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])  # 计算ap
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5
    # 计算F1(精确率和查全率的调和平均值)
    f1 = 2 * p * r / (p + r + 1e-16)  # 加上1e-16为了避免分母为0
    # 画图
    if plot:
        draw_pr_curve(px, py, ap, os.path.join(save_dir,'PR_curve.png'), names)
        draw_mc_curve(px, f1, os.path.join(save_dir, 'F1_curve.png'), names, ylabel='F1')
        draw_mc_curve(px, p, os.path.join(save_dir, 'P_curve.png'), names, ylabel='Precision')
        draw_mc_curve(px, r, os.path.join(save_dir, 'R_curve.png'), names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算box1[1,4]和box2[n,4]的IOU值
    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in (x1, y1, x2, y2) format.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # 获取box坐标
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 交叉区域面积
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # 合并的面积
    union = w1 * h1 + w2 * h2 - inter + eps
    # iou值
    iou = inter / union
    # 需要使用哪一种iou
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # # 最小包围框的w
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # 最小包围框的h
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # 包围框的斜边平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # 两个box中心距离的平方
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # 最小包围框的面积
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        # box的shape为[4,N]
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # 计算重合面积，inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou值
    return inter / (area1[:, None] + area2 - inter)