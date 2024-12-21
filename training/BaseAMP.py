import torch
from tqdm import tqdm
from torch.cuda import amp
# --------本地导入--------
from training.Base import BaseTrain


class BaseAMPTrain(BaseTrain):
    def __init__(self, param):
        super(BaseAMPTrain, self).__init__()
        self.use_amp = param["AMP"]
        # 自动混合精度，只能在cuda上使用
        self.scaler = amp.GradScaler(enabled=True) if self.use_amp else amp.GradScaler(enabled=False)
        # 设置打印抬头
        self.title_string = ("%10s" * 4) % ("Epoch", "mem", "loss-avg", "detail")

    def __call__(self, net, train_loader, optimizer, device, epoch, total_epoch, RANK, loss_function, logger, writer):
        net.train()

        logger.info(self.title_string)
        batch_avg_loss = 0.0

        # 只在主进程创建进度条
        pbar = enumerate(train_loader)
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=len(train_loader))

        # 每个batch进行迭代
        optimizer.zero_grad()
        for batch_index, (images, labels, shapes) in pbar:
            images = images.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) else labels

            # 推理
            with amp.autocast(enabled=self.use_amp):
                outputs = net(images)
                # 计算loss
                total_loss, loss_item = loss_function(outputs, labels)
            # 自动混合精度
            self.scaler.scale(total_loss).backward()  # scaler实现的反向误差传播
            self.scaler.step(optimizer)  # 优化器中的值也需要放缩
            self.scaler.update()  # 更新scaler
            optimizer.zero_grad()
            # 打印
            batch_avg_loss = (batch_avg_loss * batch_index + total_loss.item()) / (batch_index + 1)
            if RANK in [-1, 0]:
                gpu_mem = "{:.3g}G".format(torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                pbar.set_description(('%10s' * 2 + '%10.4g' + '%10.4g' * len(loss_item)) % (
                    f"{epoch}/{total_epoch - 1}", gpu_mem, batch_avg_loss, *loss_item))

        # 记录lr、loss
        if writer:
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Train/Average_loss', batch_avg_loss, epoch)

        pbar.close()