import torch
from tqdm import tqdm
# --------本地导入--------
from training.Base import BaseTrain


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count


class EpochTrain(BaseTrain):
    def __init__(self, param):
        super(EpochTrain, self).__init__()
        # 设置打印抬头
        self.title_string = ("%10s" * 4) % ("Epoch", "mem", "loss-avg", "loss")

    def __call__(self, net, train_loader, optimizer, device, epoch, total_epoch, RANK, loss_function, logger, writer):
        net.train()

        logger.info(self.title_string)
        loss_meter = AvgMeter()

        # 只在主进程创建进度条
        pbar = enumerate(train_loader)
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=len(train_loader))

        # 每个batch进行迭代
        for batch_index, batch_data in pbar:
            input_data = {key: value.to(device) for key, value in batch_data.items() if key != "caption"}
            # 推理
            output = net(input_data)
            # 计算loss
            total_loss, loss_item = loss_function(output)
            optimizer.zero_grad()
            # 反向误差传播
            total_loss.backward()
            optimizer.step()
            # 打印
            loss_meter.update(loss_item[0], input_data["image"].size(0))
            if RANK in [-1, 0]:
                gpu_mem = "{:.3g}G".format(torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                pbar.set_description(('%10s' * 2 + '%10.4g' + '%10.4g' * len(loss_item)) % (
                    f"{epoch}/{total_epoch - 1}", gpu_mem, loss_meter.avg, *loss_item))

        # 记录lr、loss
        if writer:
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Train/Average_loss', loss_meter.avg, epoch)

        pbar.close()