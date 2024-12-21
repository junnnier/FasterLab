import torch
from tqdm import tqdm
# --------本地导入--------
from training.Base import BaseTrain


class EpochTrain(BaseTrain):
    def __init__(self, param):
        super(EpochTrain, self).__init__()
        self.param = param

    def __call__(self, net, train_loader, optimizer, device, epoch, total_epoch, RANK, loss_function, logger, writer):
        net.train()

        logger.info(("{:>10}" * 3).format("Epoch", "mem", "loss-avg"))
        batch_avg_loss = 0.0

        # 只在主进程创建进度条
        pbar = enumerate(train_loader)
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=len(train_loader))

        # 每个batch进行迭代
        optimizer.zero_grad()
        for batch_index, (images, labels) in pbar:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            # 推理
            outputs = net(images)
            # 计算loss
            total_loss, loss_item = loss_function(outputs, labels)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # 打印
            batch_avg_loss = (batch_avg_loss * batch_index + total_loss.item()) / (batch_index + 1)
            if RANK in [-1, 0]:
                gpu_mem = "{:.3g}G".format(torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                pbar.set_description(('%10s' * 2 + '%10.4g' + '%10.4g' * len(loss_item)) % (f"{epoch}/{total_epoch - 1}", gpu_mem, batch_avg_loss, *loss_item))

        # 记录lr、loss
        if writer:
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Train/Average_loss', batch_avg_loss, epoch)

        pbar.close()