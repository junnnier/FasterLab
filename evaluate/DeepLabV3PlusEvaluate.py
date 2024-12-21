import torch
import numpy as np
from tqdm import tqdm
# --------本地导入--------


class StreamSegMetrics(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(self, results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class EvaluateFunction(object):
    def __init__(self, param):
        self.param = param
        self.metrics = StreamSegMetrics(param["num_classes"])
        self.metrics.reset()

    def __call__(self, model, data_loader, loss_func, device, config, save_dir="./"):
        test_avg_loss = 0.0
        describe = ('%10s' + '%10s') % ("class", "loss_avg")

        data_bar = tqdm(enumerate(data_loader), desc=describe, total=len(data_loader))
        # 对测试集进行验证
        for batch_index, (images, labels) in data_bar:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            # 计算loss
            if loss_func:
                total_loss, loss_item = loss_func(outputs, labels)
                test_avg_loss = loss_item[0]

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            self.metrics.update(targets, preds)

        score = self.metrics.get_results()
        return score['Mean IoU'], test_avg_loss

