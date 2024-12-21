import torch
import torch.nn as nn
import numpy as np


class LossFunction(object):
    def __init__(self, model, param):
        self.param = param
        self.loss_fun = nn.CrossEntropyLoss(reduction="mean")

    def __call__(self,output,target):
        loss_item = torch.zeros(1, device=output.device)
        # one-hot编码转换为classes索引
        target_device = target.device
        target = target.cpu().numpy()
        target = np.where(target == 1)[1]
        target = torch.from_numpy(target).long().to(target_device)
        total_loss = self.loss_fun(output, target)
        return total_loss, loss_item + total_loss