import torch
from torch import nn
import torch.nn.functional as F
from extension.detr_module.backbone import build_backbone
from extension.detr_module.transformer import build_transformer
from extension.detr_module.tools import NestedTensor, nested_tensor_from_tensor_list
from extension.detr_module.segmentation import DETRsegm


class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """
        Parameters:
            backbone: backbone模块。See backbone.py
            transformer: transformer模块结构。See transformer.py
            num_classes: 类别数量
            num_queries: 目标查询数，即检测槽位。即DETR可以在单个图像中检测到的最大目标数量。
            aux_loss: 如果要使用辅助解码损失(每个解码器层的损失)，则为True。
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # 用于分类
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 用于box回归
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # 用于将backbone的输出调整到和transfrom输入的维度一样，即位置编码的维度一样
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ 期望得到一个NestedTensor，包含:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: shape为[batch_size x H x W]的二进制mask, 在像素上填充1
            返回一个包含以下元素的字典:
               - "pred_logits": 所有查询的分类逻辑(包括无对象)。Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": 所有查询的规范化方框坐标表示为(center_x, center_y, height, width). 相对于每个单独图像的大小(忽略可能的填充)。
               - "aux_outputs": 可选，仅在激活辅助损失时返回。它是一个字典列表，其中包含每个解码器层的上述两个key。
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()  # 从NestedTensor对象中提取出tensors和mask
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # 是否使用辅助解码损失
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    # 辅助损失
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    """简单的多层感知器(称为FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def create_model(param):
    class_num = param["category"]
    backbone = build_backbone(param["backbone"])
    transformer = build_transformer(param["transformer"])
    model = DETR(backbone, transformer, num_classes=class_num, num_queries=param["num_queries"], aux_loss=param["aux_loss"])
    if param["masks"]:
        model = DETRsegm(model, freeze_detr=(param["frozen_weights"] is not None))
    return model


if __name__ == '__main__':
    pass