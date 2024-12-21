import torch
import torch.nn as nn
# --------本地导入--------
from utils.torch_tools import is_parallel
from extension.yolov5_module.tools import bbox_iou


class LossFunction(object):
    # 计算损失（分类损失+置信度损失+坐标框损失）
    def __init__(self, model, param, autobalance=False):
        self.param = param
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # 获取设备
        # 定义损失评判标准，nn.BCEWithLogitsLoss相当于是在nn.BCELoss()中预测结果pred的基础上先做了个sigmoid
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([param['cls_pw']], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([param['obj_pw']], device=device))

        # 标签平滑 cp(category positive), cn(category negative) 参考https://arxiv.org/pdf/1902.04103.pdf
        self.cp, self.cn = 1.0 - param.get("label_smooth", 0.0), param.get("label_smooth", 0.0)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # 提取Detect模块
        self.stride = det.stride
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7特征图对应输出的损失系数
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.gr = 1.0
        self.autobalance = autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, predict, target):
        device = target.device  # 获取设备
        # 用于记录类别损失，box损失，目标得分损失
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # 获得3个检测头对应的gt box的类别、gt box的xywh(x,y已经减去了预测方格的整数坐标)、索引、gt box所对应的anchors
        tcls, tbox, indices, anchors = self.build_targets(predict, target)

        # 遍历每个预测层，计算loss
        for i, pi in enumerate(predict):
            b, a, gj, gi = indices[i]  # 根据indices获取索引，找到对应网格的输出image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # 预测框个数
            if n:
                ps = pi[b, a, gj, gi]  # 找到对应网格的输出，取出对应位置的预测值

                # 对输出的xywh做反算
                pxy = ps[:, :2].sigmoid() * 2. - 0.5  # 将预测的中心点坐标变换到-0.5到1.5之间
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # 预测的box
                iou = bbox_iou(pbox, tbox[i], xywh=True, CIoU=True).squeeze(dim=1)  # 计算ciou(prediction, target)
                lbox += (1.0 - iou).mean()  # 最终iou loss
                score_iou = iou.detach().clamp(0).type(tobj.dtype)  # detach函数使得iou不可反向传播，clamp将小于0的iou裁剪为0
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)  # 返回排序后的索引值
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                # 根据model.gr设置object的标签值；因为不同anchor和gt bbox匹配度不一样，预测框和gt bbox匹配度也不一样
                # 将预测框和bbox的iou作为权重乘到conf分支，用于表征预测质量
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # 类别数大于1才计算分类损失
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # ps[:, 5:]取num个预测框信息的第6个开始后面的数据，即目标是每个类别的概率。
                    t[range(n), tcls[i]] = self.cp  # self.cn和self.cp分别是标签平滑的负样本平滑标签和正样本平滑标签
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.param['box']  # 乘上超参里设置的权重
        lobj *= self.param['obj']
        lcls *= self.param['cls']
        bs = tobj.shape[0]  # batch size大小

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    # 构建训练时的target (image,class,x,y,w,h)
    def build_targets(self, p, targets):

        """
        p:网络输出，是一个list，有3个tensor元素，每个tensor的shape是（b,3,h,w,class+4(x,y,w,h)+1(是否为前景))，其中3是每个网格有3个锚框
        targets的shape为（nt，6），其中6是icxywh，i是第i+1张图片，c是目标类别，坐标xywh
        """
        na, nt = self.na, targets.shape[0]  # na: anchor的数量, nt: gt box的数量（一个batch中的）
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # 初始化网格[1, 1, 1, 1, 1, 1, 1]
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # anchor索引，shape是[3,gt box数量]，第一行全是0，第2行全是1，第三行全是2，用于表示当前bbox和当前层的哪个anchor匹配。
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # shape是[3,gt box数量,7]，先将targets复制3次，对应着当前层的三种anchor，因为不确定哪个anchor是最合适的。原本target是6列，第7列是给每个gt box加上索引，表示对应着哪种anchor

        g = 0.5  # 网格中心偏移量
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m 对应上下左右四个网格的偏移量
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm 对应四个角的网格
                            ], device=targets.device).float() * g  # offsets
        # 对每个检测层进行处理
        for i in range(self.nl):
            anchors = self.anchors[i]  # 当前分支的anchor，shape是[3,2]（已经除以了当前特征图对应的stride）
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # 当前特征图大小[1,1,80,80,80,80,1]，80的位置对应着gt box信息的xywh

            # 将目标与锚点匹配
            t = targets * gain  # targets里的xywh是归一化到0 ~ 1之间的，乘以gain之后，将targets的xywh映射到检测头的特征图大小上。
            if nt:
                # 匹配，wh回归方式(wh.sigmoid()*2)**2*anchors[i]，倍数控制在0~4之间
                r = t[:, :, 4:6] / anchors[:, None]  # 计算wh的比例，r的shape为[3,gt box数量,2]，2分别表示gt box的w和h与anchor的w和h的比值。
                j = torch.max(r, 1. / r).max(2)[0] < self.param['anchor_t']  # j的shape为(3, gt_box数量)，进行比较，筛选1/hyp["anchor_t"] < 比值 < hyp["anchort_t"]的框。过滤的原因是和anchor的宽高差别较大的gt box是非常难预测的，不适合用来训练
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # 过滤

                # 寻找网格附近的框
                gxy = t[:, 2:4]  # 取出过滤后的gt box的中心点浮点型的坐标。
                gxi = gain[[2, 3]] - gxy  # 将以图像左上角为原点的坐标变换为以图像右下角为原点的坐标。
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T  # 浮点数取模：a%b=a-(a/b)*b，以图像左上角为原点的坐标，也就是取中心点的小数部分，小数部分小于0.5的为ture，大于0.5的为false，true的位置分别表示靠近方格左边的gt box和靠近方格上方的gt box。也就是找出邻近的网格，将这些网格都认为是负责预测该bbox的网格。>1是因为只有中心点在最外圈网格内才有上面和左边的方格。
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T  # 以图像右下角为原点的坐标，取中心点的小数部分，小数部分小于0.5的为ture，大于0.5的为false。true的位置分别表示靠近方格右边的gt box和靠近方格下方的gt box。
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # 将j, k, l, m组合成一个tensor，也就是上下左右4个网格，另外还增加了一个全为true的维度用于表示自身网格。shape是[5,过滤后的gt_box数量]
                t = t.repeat((5, 1, 1))[j]  # 将t复制5次，对应着5种网格进行筛选
                """
                上面这一步将t复制5个，然后使用j来过滤，
                第一个t是保留所有的gt box，也就是自身方格的gt box，因为上一步里面增加了一个全为true的维度，
                第二个t保留了靠近方格左边的gt box，
                第三个t保留了靠近方格上方的gt box，
                第四个t保留了靠近方格右边的gt box，
                第五个t保留了靠近方格下边的gt box，
                """
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # shape为(num,2),表示保留下来的num个gt box的x,y对应的偏移，一个gt box在以上五个t里面，只会有三个t是true。
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # 对应的image和class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()  # 当前的label落在哪个网格上，将中心点偏移到相邻最近的方格里，然后向下取整
            gi, gj = gij.T  # 网格的索引

            # Append
            a = t[:, 6].long()  # 对应anchor的索引
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        """
        tcls的shape为(3, num), 表示3个检测头对应的gt box的类别。

        tbox的shape为(3, ([num, 4]))， 表示3个检测头对应的gt box的xywh， 其中x和y已经减去了预测方格的整数坐标，
        比如原始的gt box的中心坐标是(51.7, 44.8)，则该gt box由方格(51, 44)，以及离中心点最近的两个方格(51, 45)和(52, 44)来预测,
        换句话说这三个方格预测的gt box是同一个，其中心点是(51.7, 44.8)，但tbox保存这三个方格预测的gt box的xy时，保存的是针对这三个方格的偏移量,
        分别是：
        (51.7 - 51 = 0.7, 44.8 - 44 = 0.8)
        (51.7 - 51 = 0.7, 44.8 - 45 = -0.2)
        (51.7 - 52 = -0.3, 44.8 - 44 = 0.8)

        indices的shape为(3, ([num], [num], [num], [num])), 4个num分别表示每个gt box(包括偏移后的gt box)在batch中的image index， anchor index， 预测该gt box的网格y坐标， 预测该gt box的网格x坐标。

        anchors的shape为(3, ([num, 2]))， 表示每个检测头对应的num个gt box所对应的anchor。
        """
        return tcls, tbox, indices, anch