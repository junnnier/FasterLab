import itertools
from utils.torch_tools import time_sync


class PostProcess(object):
    def __init__(self):
        char_file = "/mnt/work01/project/wantianjun/general_detection/other_dataset/Synthetic_Chinese/char_std_5990.txt"
        with open(char_file, "r", encoding="utf-8") as f:
            self.char_index_list = [item.strip() for item in f.readlines()]

    def __call__(self, output, shapes, config, image_path, save):
        st = time_sync()
        pre_index_list = self.ctc_decode(output)
        et = time_sync() - st
        pre_string = [self.char_index_list[index] for index in pre_index_list[0]]
        # 打印
        result_info = "{}\t{}".format(image_path, "".join(pre_string))
        print(result_info)
        return result_info, et

    def ctc_decode(self, pred, blank_index=0):  # T * N * C
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