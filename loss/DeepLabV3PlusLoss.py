import torch.nn as nn
import torch.nn.functional as F
import torch


class LossFunction(nn.Module):
    def __init__(self, model, param):
        super(LossFunction, self).__init__()
        self.alpha = param["alpha"]
        self.gamma = param["gamma"]
        self.ignore_index = param["ignore_index"]
        self.size_average = param["size_average"]

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean(), [focal_loss.mean().item()]
        else:
            return focal_loss.sum(), [focal_loss.mean().item()]