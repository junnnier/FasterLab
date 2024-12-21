import math
from tqdm import tqdm


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count


class EvaluateFunction(object):
    def __init__(self, param):
        self.param = param

    def __call__(self, model, data_loader, loss_func, device, config, save_dir=""):
        loss_meter = AvgMeter()
        describe = ('%10s') % ("avg_loss")

        data_bar = tqdm(enumerate(data_loader), desc=describe, total=len(data_loader))
        # 对验证集进行预测
        for batch_index, batch_data in data_bar:
            input_data = {key: value.to(device) for key, value in batch_data.items() if key != "caption"}
            # 推理
            output = model(input_data)
            # 计算loss
            if loss_func:
                total_loss, loss_item = loss_func(output)
                loss_meter.update(loss_item[0], input_data["image"].size(0))
        data_bar.close()

        # 打印结果
        accuracy = - math.log(loss_meter.avg * 0.1) / 100
        print(("%10.4g") % (accuracy))

        return accuracy, loss_meter.avg