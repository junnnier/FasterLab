import torch
import torch.nn as nn


class LossFunction(object):
    def __init__(self, model, param):
        self.param = param
        self.loss_fun = nn.BCELoss(reduction="mean")

    def __call__(self,output,target):
        loss_item = torch.zeros(1, device=output.device)
        total_loss = self.loss_fun(output, target)
        return total_loss, loss_item + total_loss