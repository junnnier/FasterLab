import torch
from tqdm import tqdm
# --------本地导入--------
from training.Base import BaseTrain


class EpochTrain(BaseTrain):
    def __init__(self, param):
        super(EpochTrain, self).__init__()
        # 设置打印抬头
        self.title_string = ("%10s" * 4) % ("Epoch", "mem", "loss-avg", "ctc-loss")

    def __call__(self, net, train_loader, optimizer, device, epoch, total_epoch, RANK, loss_function, logger, writer):
        net.train()

        logger.info(self.title_string)
        batch_avg_loss = 0.0

        # 只在主进程创建进度条
        pbar = enumerate(train_loader)
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=len(train_loader))

        # 每个batch进行迭代
        for batch_index, batch_data in pbar:
            images, labels, labels_length = batch_data["image"], batch_data["label"], batch_data["label_length"]
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)
            # 推理
            output = net(images)
            log_preds = output.log_softmax(dim=2)
            input_lengths = torch.tensor([len(item) for item in output.permute(1, 0, 2)], dtype=torch.long, device=device)
            target_lengths = torch.tensor(labels_length, device=device, dtype=torch.long)
            # 计算loss
            total_loss, loss_item = loss_function(log_preds, labels, input_lengths, target_lengths)
            batch_avg_loss = (batch_avg_loss * batch_index + loss_item[0] * len(batch_data)) / (batch_index + 1)

            optimizer.zero_grad()
            # 反向误差传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=0.1)  # 防止梯度爆炸
            optimizer.step()
            # 打印
            if RANK in [-1, 0]:
                gpu_mem = "{:.3g}G".format(torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                pbar.set_description(('%10s' * 2 + '%10.4g' + '%10.4g' * len(loss_item)) % (
                    f"{epoch}/{total_epoch - 1}", gpu_mem, batch_avg_loss, *loss_item))

        # 记录lr、loss
        if writer:
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Train/Average_loss', batch_avg_loss, epoch)

        pbar.close()