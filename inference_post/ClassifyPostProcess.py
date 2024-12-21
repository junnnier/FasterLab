from utils.torch_tools import time_sync


class PostProcess(object):
    def __init__(self):
        pass

    def __call__(self, output, shapes, config, image_path, save):
        label_index_name = config["LABEL_INDEX_NAME"]
        st = time_sync()
        confidence, pred_index = output.max(1)
        et = time_sync() - st
        # 打印
        result_info = "{}\t{}\t{:.6f}".format(image_path, label_index_name[pred_index], confidence.item())
        print(result_info)
        return result_info, et