import torch.nn as nn
from extension.SSIM import SSIM


class LossFunction(nn.Module):
    def __init__(self, model, param):
        super(LossFunction, self).__init__()
        self.param = param
        self.criterion_L1 = nn.L1Loss()
        self.criterion_ssim = SSIM()

    def __call__(self, output, target):
        ssim_loss = - self.criterion_ssim(target, output)
        Pixellevel_L1_Loss = self.criterion_L1(output, target)
        total_loss = Pixellevel_L1_Loss + 0.2 * ssim_loss
        return total_loss, [Pixellevel_L1_Loss.item(), ssim_loss.item()]