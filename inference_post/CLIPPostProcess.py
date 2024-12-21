import torch
from utils.torch_tools import time_sync


class PostProcess(object):
    def __init__(self):
        self.label_index_name = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

    def __call__(self, output, shapes, config, image_path, save):
        st = time_sync()
        image_embeddings, text_embeddings = output
        similarity = torch.matmul(image_embeddings, text_embeddings.T).squeeze(0)
        predicted_class = similarity.argmax().item()
        et = time_sync() - st
        # 打印
        result_info = "{}\t{}\t{:.6f}".format(image_path, self.label_index_name[predicted_class], similarity[predicted_class])
        print(result_info)
        return result_info, et