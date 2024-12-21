import numpy as np
import torch
import cv2
# --------本地导入--------
from utils.image_tools import load_image, resize_image_tough


class PreTreatment(object):
    def __init__(self):
        pass

    def __call__(self, image_path, image_size, force_resize, device):
        # 读取图片
        image = load_image(image_path)
        image, ori_img_shape, scale, fill_w_h_dist = resize_image_tough(image, image_size)
        image_shapes = (ori_img_shape, scale, fill_w_h_dist)
        # 转化为灰度图，零均值化
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image * 255.0
        image = image / 255.0 * 2.0 - 1.0
        # 灰度图是二维的，判断是否需要增加一个通道
        if len(image.shape) < 3:
            image = image[:, :, np.newaxis]
        # 转换
        image = image.transpose((2, 0, 1))  # HWC to CHW
        image = np.ascontiguousarray(image,dtype=np.float32)  # 转换为内存连续存储的数组
        image = torch.from_numpy(image).to(device)
        image = torch.unsqueeze(image, dim=0)  # 增加一个维度
        return image, image_shapes