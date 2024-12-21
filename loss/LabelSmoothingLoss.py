import torch
import torch.nn as nn


class LossFunction(nn.Module):
    def __init__(self, model, param, dim=-1):
        super(LossFunction, self).__init__()
        self.param = param
        self.dim = dim

    def forward(self, pred, target):
        total_loss = torch.mean(torch.sum(-target * torch.log(pred), dim=self.dim))
        return total_loss, [total_loss.item()]