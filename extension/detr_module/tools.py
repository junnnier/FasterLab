from typing import Optional
from torch import Tensor
import torch
from scipy.optimize import linear_sum_assignment  # 匈牙利算法
from torch import nn
import torchvision
import torch.nn.functional as F
from torchvision.ops.boxes import box_area
import torch.distributed as dist
from packaging import version
if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    box为[x0, y0, x1, y1]格式，返回一个[N, M]成对的矩阵, 其中 N = len(boxes1) 且 M = len(boxes2)
    """
    # 防止boxes出现 inf / nan 的结果
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            return _onnx_nested_tensor_from_tensor_list(tensor_list)
        # 适应不同大小的图片，从所有图片中找到最大的w,h,c，进行填补
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)  # 用于记录图片有效区域的mask
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)
        # 可以支持在onnx中工作
        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


class HungarianMatcher(nn.Module):
    # 计算目标与网络预测之间的匹配。预测通常多于目标。在这种情况下，我们对最佳预测进行一对一的匹配，而其他的则不匹配。
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """
        Params:
            cost_class: 分类误差的相对权重
            cost_bbox: 边界框坐标L1误差的相对权重
            cost_giou: 边界框损失的相对权重
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": 预测类别逻辑shape为[batch_size, num_queries, num_classes]的tensor
                 "pred_boxes": 预测的方框坐标shape为[batch_size, num_queries, 4]的tensor

            targets: 这是一个目标列表(len(targets) = batch_size)，其中每个目标是一个字典，包含:
                 "labels": shape为[num_target_boxes]的tensor(其中num_target_boxes是目标中基本真实对象的数量)包含类标签
                 "boxes": shape为[num_target_boxes, 4]包含目标框坐标的tensor
        Returns:
            一个大小为batch_size的列表，包含(index_i, index_j)元组，其中:
                - index_i 是所选预测的索引(按顺序)
                - index_j 为所选目标对应的索引(按顺序)
            每个batch的元素，是不变的:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 在batch中用平面化的方法来计算批量的矩阵
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # 提取预测类别[batch_size * 100, 92]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # 提取预测框[batch_size * 100, 4]

        # 连接目标标签和框
        tgt_ids = torch.cat([v["labels"] for v in targets])  # 获取batch中所有的类别
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # 获取所有的框

        # 计算分类成本，近似于 1 - proba[target class].
        cost_class = -out_prob[:, tgt_ids]  # 从预测出的类别中，提取出标注了类别的预测结果

        # 计算预测的box和目标box之间的L1损失
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 计算预测的box和目标box之间的之间的giou损失
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # 最终的矩阵损失
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()  # 还原回[batch_size, 100, n]这里n是batch中所有的类别数量

        sizes = [len(v["boxes"]) for v in targets]  # 得到每个图中box的个数

        # 按box个数从最终的矩阵损失中，找到对应batch中的结果并分割出相应的loss，然后使用匈牙利算法，找到和最小的组合的索引（row、col）
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SetCriterion(nn.Module):
    """ 
    计算DETR的loss，过程分为两个步骤：
        1) 计算ground truth boxes与模型输出之间的匈牙利匹配
        2) 监督每一对匹配的ground-truth/prediction(监督类和框)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """
        Args:
            num_classes: 对象类别的数目，省略特殊的无对象类别
            matcher: 能够计算目标和建议框之间的匹配模块
            weight_dict: 一个额字典，包含损失的名称及其相对权重的值。
            eos_coef: 无对象类别的相对分类权重
            losses: 适用的所有损失列表
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        分类loss，targets字典必须包含包含一个维度为[nb_target_boxes]的tensor"
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # 提取出类别信息，[2, 100, 92]

        idx = self._get_src_permutation_idx(indices)  # 获取最优匹配关系的索引，有两个元素（第几张图，匹配到的第几个预测框）
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # 根据匹配到的框，获取真实的目标类别
        # 创建一个填充均为背景类别的[bs,100]矩阵，并把目标类别填充到对应的位置
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        # 计算交叉商
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        # 是否单独记录一下错误类别的精度
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        计算基数误差，即预测的非空框数量的绝对误差。这并不是真正的损失，它只是用于记录目的。不会进行梯度传播
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
           计算与boxes相关的损失、L1回归损失和GIoU损失。目标字典必须包含键"boxes"，其中包含一个tensor的shape为[nb_target_boxes, 4]，
           目标框的格式为(center_x, center_y, w, h)，值为图像大小归一化后的。
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)  # 获取最优匹配关系的索引，有两个元素（第几张图，匹配到的第几个预测框）
        src_boxes = outputs['pred_boxes'][idx]  # 提取出预测框的坐标，总共有N个框[N, 4]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # 获取真实框的坐标，总共有N个框[N, 4]
        # 计算L1损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        # 计算L1损失的平均值
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        # 计算GIOU损失，取平均值
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))  # 计算GIOU损失
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # 按索引排列出batch的预测
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """
        Parameters:
             outputs: 模型输出，一个tensor字典
             targets: 字典列表, such that len(targets) == batch_size
        """
        # 提取出最后一层输出
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # 最后一层的输出与目标之间的最佳匹配索引
        indices = self.matcher(outputs_without_aux, targets)

        # 为了规范化，计算所有节点上目标框的平均数量
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # 计算所有请求的主网络的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 如果有辅助损失的情况下，对每个中间层的输出重复这个计算过程。
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    # 中间计算mask损失太费时，所以忽略它们。
                    if loss == 'masks':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # 只记录最后一层
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class PostProcess(nn.Module):
    """
    将模型的输出转换为coco api所期望的格式
    """
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: 模型的原始输出
            target_sizes: 维度为[batch_size x 2]的张量，包含批处理中每个图像的大小。为了进行评估，这必须是原始图像大小(在任何数据增强之前)。
            对于可视化，这应该是数据增强之后，填充之前的图像大小
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # 转换为[x0, y0, x1, y1]格式
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # 从相对坐标[0,1]到绝对坐标[0,1]
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this class can go away.
    """
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes