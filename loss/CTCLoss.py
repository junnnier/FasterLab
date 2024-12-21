from torch.nn import CTCLoss


class LossFunction(object):
    def __init__(self, model, param):
        self.param = param
        self.loss_fun = CTCLoss(blank=param["blank"])

    def __call__(self, log_preds, labels, input_lengths, target_lengths):
        loss = self.loss_fun(log_preds, labels, input_lengths, target_lengths)
        return loss, [loss.item()]