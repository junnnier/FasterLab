import os
from tqdm import tqdm
from sklearn.metrics import classification_report


class EvaluateFunction(object):
    def __init__(self, param):
        self.param = param

    def __call__(self, model, data_loader, loss_func, device, config, save_dir=""):
        test_avg_loss = 0.0
        total_top1_num = 0.0
        total_top5_num = 0.0
        outputs_list = []
        labels_list = []
        describe = ('%10s' + '%10s' * 2) % ("class", "Top1", "Top5")

        data_bar = tqdm(enumerate(data_loader), desc=describe, total=len(data_loader))
        # 对验证集进行预测
        for batch_index, (images, labels, shapes) in data_bar:
            images = images.to(device)
            labels = labels.to(device)
            # 推理
            outputs = model(images)
            # 计算loss
            if loss_func:
                total_loss, loss_item = loss_func(outputs, labels)
                test_avg_loss = (test_avg_loss * batch_index + total_loss.item()) / (batch_index + 1)
            # 统计结果
            topk_num, o_index, l_index = self.count_accuracy(outputs,labels)
            total_top1_num += topk_num[0]
            total_top5_num += topk_num[1]
            outputs_list.extend(o_index)
            labels_list.extend(l_index)
        data_bar.close()

        # 打印结果
        accuracy_top1 = total_top1_num / len(data_loader.dataset)
        accuracy_top5 = total_top5_num / len(data_loader.dataset)
        print(("%10s" + "%10.3g" * 2) % ("all", accuracy_top1, accuracy_top5))

        if save_dir:
            measure_result = classification_report(labels_list, outputs_list, target_names=config["LABEL_INDEX_NAME"])
            print("measure_result: \n")
            print(measure_result)
            result_save_path = os.path.join(save_dir, "val_result.txt")
            with open(result_save_path, "w", encoding="utf-8") as f:
                f.write("accuracy top1: {}\taccuracy top5: {}\n".format(accuracy_top1, accuracy_top5))
                f.write(measure_result)
            print("The validation result save to:", result_save_path)

        return accuracy_top1, test_avg_loss

    @staticmethod
    def count_accuracy(output, labels, topk=(1, 5)):
        """
        统计topk的正确数量
        Args:
            output: shape为[N, class_num]
            labels: one-hot编码，shape为[N, class_num]
            topk: top k
        Returns:
        """
        topk_result = []
        l_data, l_index = labels.data.max(1, keepdim=True)
        # 按最大到小排序，返回最大的k个结果。
        o_data, o_index = output.topk(max(topk), dim=1, largest=True, sorted=True)
        o_index = o_index.t()  # 从[N, k]到[k, N]
        # 预测正确的索引
        correct_mask = o_index.eq(l_index.view(1, -1).expand_as(o_index))
        # 统计topk正确数量
        for k in topk:
            correct_k = correct_mask[:k].reshape(-1).float().sum(0, keepdim=True).item()
            topk_result.append(correct_k)

        # 提取top1的索引
        o_index = o_index[:1].reshape(-1).cpu().numpy().tolist()
        l_index = l_index.reshape(-1).cpu().numpy().tolist()
        return topk_result, o_index, l_index