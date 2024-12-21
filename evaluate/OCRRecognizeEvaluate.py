import itertools
from tqdm import tqdm


class EvaluateFunction(object):
    def __init__(self, param):
        self.param = param

    def __call__(self, model, data_loader, loss_func, device, config, save_dir=""):
        test_avg_loss = 0.0
        test_number = 0
        acc = 0.0
        describe = ('%10s' + '%10s') % ("number", "accuracy")

        data_bar = tqdm(enumerate(data_loader), desc=describe, total=len(data_loader))
        # 对验证集进行预测
        for batch_index, batch_data in data_bar:
            images, labels, labels_length = batch_data["image"], batch_data["label"], batch_data["label_length"]
            images = images.to(device)
            # 推理
            outputs = model(images)
            # 计算loss
            if loss_func:
                pass
            # 统计结果
            pred_labels = self.ctc_decode(outputs)
            for p, gt, l in zip(pred_labels, labels.numpy(), labels_length.numpy()):
                # 必须全部匹配才算正确
                if p == gt[:int(l)].tolist():
                    correct = 1
                else:
                    correct = 0
                acc = (acc * test_number + correct) / (test_number + 1)
                test_number += 1
        data_bar.close()

        # 打印结果
        print(("%10s" + "%10.3g") % (test_number, acc))
        return acc, test_avg_loss

    @staticmethod
    def ctc_decode(pred, blank_index=0):  # T * N * C
        arg_max = pred.argmax(dim=-1)  # T * N
        arg_max = arg_max.t()  # N * T
        arg_max = arg_max.to(device='cpu').numpy()
        pred_labels = []
        for line in arg_max:
            label = [k for k, g in itertools.groupby(line)]
            while blank_index in label:
                label.remove(blank_index)
            pred_labels.append(label)
        return pred_labels