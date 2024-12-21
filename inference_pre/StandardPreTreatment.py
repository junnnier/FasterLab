import numpy as np
import torch
# --------本地导入--------
from utils.image_tools import load_image, resize_image, resize_image_tough


class PreTreatment(object):
    def __init__(self):
        pass

    def __call__(self, image_path, image_size, force_resize, device):
        # 读取图片
        image = load_image(image_path)
        image, ori_img_shape, scale, fill_w_h_dist = resize_image_tough(image, image_size) if force_resize else resize_image(image, image_size[0])
        image_shapes = (ori_img_shape, scale, fill_w_h_dist)
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image, dtype=np.float32)  # 转换为内存连续存储的数组
        image = torch.from_numpy(image).to(device) / 255.0  # 归一化
        image = torch.unsqueeze(image, dim=0)  # 增加一个维度
        return image, image_shapes