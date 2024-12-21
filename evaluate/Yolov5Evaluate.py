import numpy as np
from tqdm import tqdm
import torch
# --------本地导入--------
from extension.yolov5_module.tools import ConfusionMatrix, non_max_suppression, process_batch, ap_per_class, accuracy_fitness
from utils.label_tools import xywh2xyxy, coordinate_scale, clip_coords


class EvaluateFunction(object):
    def __init__(self, param):
        self.conf_thres = param["conf_thres"]  # 置信度阈值
        self.iou_thres = param["iou_thres"]  # NMS的IoU阈值
        self.single_cls = param["single_cls"]  # 全部视为单一类数据集
        self.plots = param["plots"]  # 画图

    def __call__(self, model, data_loader, loss_func, device, config, save_dir="./"):

        device = next(model.parameters()).device  # 获取模型所在device

        # 是否使用半精度，只支持在CUDA上
        half = False  # device.type != "cpu"
        model.half() if half else model.float()

        # 初始配置
        nc = len(config["LABEL_INDEX_NAME"])  # 类别数量
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # mAP@0.5到0.95的IOU向量
        niou = iouv.numel()  # 总共有多少个元素
        seen = 0  # 统计验证过的图片数量
        confusion_matrix = ConfusionMatrix(nc=nc)  # 创建混淆矩阵
        names = {k: v for k, v in enumerate(config["LABEL_INDEX_NAME"])}
        p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        total_loss = torch.zeros(3, device=device)  # 总损失：box, obj, cls
        stats, ap, ap_class = [], [], []
        # 对测试集进行验证
        describe = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        for batch_i, (img, targets, shapes) in enumerate(tqdm(data_loader, desc=describe)):
            # 预处理
            img = img.to(device)
            img = img.half() if half else img.float()  # 从uint8到fp16/32
            targets = targets.to(device)

            # 推理
            out, train_out = model(img)

            # 是否计算loss
            if loss_func:
                total_loss += loss_func([x.float() for x in train_out], targets)[1]

            # 计算NMS, 返回一个list，每个元素是一张图的结果，shape为[N,6], 其中6为[x, y, x, y, conf, cls]
            _, _, height, width = img.shape  # 输入图片的宽、高
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # 恢复到输入图片的像素坐标位置
            out = non_max_suppression(out, self.conf_thres, self.iou_thres, multi_label=True, agnostic=self.single_cls)

            # 统计每张图片
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]  # 提取第si张图的标签信息
                nl = len(labels)  # 框的数量
                tcls = labels[:, 0].tolist() if nl else []  # 框的类别
                seen += 1

                # 没有预测到框
                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # 全部视为单一类数据集
                if self.single_cls:
                    pred[:, 5] = 0

                # 预测坐标恢复到原图位置上
                predn = pred.clone()
                coordinate_scale(predn[:, :4], 1/shapes[si][1], -shapes[si][2][0], -shapes[si][2][1])
                clip_coords(predn[:, :4], shapes[si][0])

                # 验证
                if nl:
                    labels[:, 1:5] = xywh2xyxy(labels[:, 1:5])  # target boxes
                    coordinate_scale(labels[:, 1:5], 1/shapes[si][1], -shapes[si][2][0], -shapes[si][2][1])
                    labelsn = labels[:, 0:5]
                    correct = process_batch(predn, labelsn, iouv)  # 各iou值下预测正确的索引矩阵，shape为[N, 10]
                    if self.plots:
                        confusion_matrix.process_batch(predn, labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)

                # 添加到统计结果(correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # 统计最终结果，计算pr曲线等
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=self.plots, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # 每个类的目标数
        else:
            nt = torch.zeros(1)

        # 打印结果
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 4
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

        # 打印每个类的结果
        if len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # 画图
        if self.plots:
            confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

        # Return results
        model.float()
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]

        # 最合适的精度，结果的加权值=0.1*mAP@.5 + 0.9*mAP@.5-.95
        acc_result = [mp, mr, map50, map]
        acc = accuracy_fitness(acc_result)

        return acc, torch.sum(total_loss/len(data_loader)).cpu().item()