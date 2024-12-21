from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import math


def StepLR(optimizer, param):
    train_scheduler = lr_scheduler.StepLR(optimizer, step_size=param["step_size"], gamma=param["gamma"])
    return train_scheduler


def MultiStepLR(optimizer, param):
    train_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=param["milestones"], gamma=param["gamma"])
    return train_scheduler


def CosineAnnealingLR(optimizer, param):
    train_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=param["T_max"], eta_min=param["eta_min"])
    return train_scheduler


def ExponentialLR(optimizer, param):
    train_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=param["gamma"])
    return train_scheduler


def LinearLR(optimizer, param):
    lf = lambda x: (1 - x / (param["step"] - 1)) * (1.0 - param['lrf']) + param['lrf']
    train_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return train_scheduler


def CosineHalfPeriodLR(optimizer, param):
    import math
    lf = lambda x: 0.5 * (1. + math.cos(math.pi * x / param["epoch"]))
    train_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return train_scheduler


def ReduceLROnPlateau(optimizer, param):
    train_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=param["patience"], factor=param["factor"])
    return train_scheduler


def DeRainLR(optimizer, param):
    lf = lambda x: param["lrf"]*(param["epoch"]-x)/(param["epoch"]-param["milestones"]) if x >= param["milestones"] else param["lrf"]
    train_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return train_scheduler


def OneCycleLR(optimizer, param):
    lf = lambda x: ((1 - math.cos(x * math.pi / param["epoch"])) / 2) * (param["lrf"] - 1) + 1
    train_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return train_scheduler


def DeepLabV3PlusLR(optimizer, param):
    total_itrs = param["total_itrs"]
    train_scheduler = PolyLR(optimizer, total_itrs)
    return train_scheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, scheduler, warmup_iters=0, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            scheduler: warmup之后的scheduler
            warmup_iters: warmup需要迭代次数
            last_epoch: 最后一次迭代的index，如果是中断继续训练就等于加载的模型的epoch。默认为-1表示从头开始训练
        """
        self.warmup_iters = warmup_iters
        self.after_scheduler = scheduler
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        # 超过warmup范围，使用传入的scheduler类，否则使用当前重构类的
        if epoch > self.warmup_iters:
            if epoch is None:
                self.after_scheduler.step()
            else:
                # self.after_scheduler.step(epoch - self.warmup_iters)  # 注意要从0个epoch开始，所以需要减去
                self.after_scheduler.step()  # 注意要从0个epoch开始，所以需要减去
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            super(WarmUpLR, self).step()

    def get_lr(self):
        # 超过warmup范围，使用传入的scheduler类的get_lr()，否则使用自定义的学习率变化
        if self.last_epoch > self.warmup_iters:
            return self.after_scheduler.get_last_lr()
        else:
            # 在前m次迭代前，修改设置学习速率为base_lr * m / total_iters，分母加上实数1e-8防止分母为0
            return [base_lr * self.last_epoch / (self.warmup_iters + 1e-8) for base_lr in self.base_lrs]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import yaml
    from utils.select_tools import get_optimizer, get_scheduler, get_network

    # 可视化学习率变化
    def show_learning_rate_curve(scheduler, epochs):
        learning_rates = []
        for _ in range(epochs):
            optimizer.step()
            learning_rates.append(scheduler.get_last_lr()[0])
            scheduler.step()
        plt.plot(list(range(epochs)), learning_rates)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title(type(scheduler).__name__)
        plt.tight_layout()
        plt.show()

    config_path = "../config/rain100H.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    net = get_network(conf).to("cpu")
    optimizer = get_optimizer(net, conf)
    train_scheduler = get_scheduler(optimizer, conf)
    if conf["WARMUP"] > 0:
        warmup_scheduler = WarmUpLR(optimizer, train_scheduler, conf["WARMUP"])
        show_learning_rate_curve(warmup_scheduler, conf["EPOCH"])  # 画图
    else:
        show_learning_rate_curve(train_scheduler, conf["EPOCH"])

