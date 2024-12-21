import torch
from tqdm import tqdm
# --------本地导入--------
from training.Base import BaseTrain
from extension.detr_module.misc import MetricLogger, SmoothedValue


class EpochTrain(BaseTrain):
    def __init__(self, param):
        super(EpochTrain, self).__init__()
        self.param = param
        self.clip_max_norm = param["clip_max_norm"]
        self.metric_logger = MetricLogger(delimiter="  ")
        self.metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        self.metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))

    def __call__(self, net, train_loader, optimizer, device, epoch, total_epoch, RANK, loss_function, logger, writer):
        net.train()
        loss_function.train()

        logger.info(("{:>10}" * 3).format("Epoch", "mem", "loss-avg"))
        batch_avg_loss = 0.0

        # 只在主进程创建进度条
        pbar = enumerate(train_loader)
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=len(train_loader))

        # 每个batch进行迭代
        optimizer.zero_grad()
        for batch_index, (images, labels) in pbar:
            images = images.to(device)
            labels = [{key: value.to(device) for key, value in label.items()} for label in labels]

            # 推理
            outputs = net(images)
            # 计算loss
            loss_dict, loss_item = loss_function(outputs, labels)
            weight_dict = loss_function.weight_dict
            total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            optimizer.zero_grad()
            total_loss.backward()
            if self.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.clip_max_norm)
            optimizer.step()
            # 打印
            batch_avg_loss = (batch_avg_loss * batch_index + total_loss.item()) / (batch_index + 1)
            if RANK in [-1, 0]:
                gpu_mem = "{:.3g}G".format(torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                pbar.set_description(('%10s' * 2 + '%10.4g') % (f"{epoch}/{total_epoch - 1}", gpu_mem, batch_avg_loss))
                pbar.set_postfix(**loss_item)

        # 记录lr、loss
        if writer:
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Train/Average_loss', batch_avg_loss, epoch)

        pbar.close()