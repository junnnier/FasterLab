from extension.detr_module.tools import HungarianMatcher, SetCriterion


class LossFunction(object):
    def __init__(self, model, param):
        device = next(model.parameters()).device  # 获取设备
        self.param = param
        # 匈牙利算法对象，用于从100个预测框中筛选出和目标框一一对应的框
        matcher = HungarianMatcher(cost_class=param["cost_class"], cost_bbox=param["cost_bbox"], cost_giou=param["cost_giou"])
        # 各个损失计算时的权重
        self.weight_dict = {'loss_ce': 1, 'loss_bbox': param["bbox_loss_coef"], 'loss_giou': param["giou_loss_coef"]}
        if param["masks"]:
            self.weight_dict["loss_mask"] = param["mask_loss_coef"]
            self.weight_dict["loss_dice"] = param["dice_loss_coef"]
        # 额外trick
        if param["aux_loss"]:
            aux_weight_dict = {}
            for i in range(param["dec_layers"] - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in self.weight_dict.items()})
            self.weight_dict.update(aux_weight_dict)
        losses = ['labels', 'boxes', 'cardinality']
        if param["masks"]:
            losses += ["masks"]
        self.criterion = SetCriterion(param["num_classes"], matcher=matcher, weight_dict=self.weight_dict, eos_coef=param["eos_coef"], losses=losses)
        self.criterion.to(device)

    def __call__(self, predict, target):
        loss_dict = self.criterion(predict, target)
        # 自定义细节
        loss_detial = {k: loss_dict[k].item() * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict}
        return loss_dict, loss_detial

    def train(self):
        self.criterion.train()

    def eval(self):
        self.criterion.eval()