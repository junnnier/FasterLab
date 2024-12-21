import random
import cv2
import numpy as np
import torch
# --------本地导入--------
from dataset.Base import BaseDataset
from utils.image_tools import load_image,resize_image_tough,ColorAugment


class DatasetTask(BaseDataset):
    def __init__(self, conf, describe=""):
        """
        Args:
            conf: 配置文件
            describe: train还是test
        """
        super(DatasetTask, self).__init__()
        self.train_mode = True if describe == "train" else False
        self.image_list = []
        self.label_list = []
        self.imgsz = conf["IMAGE_SIZE"]  # 图片尺寸
        self.color_augment = conf["COLOR_AUGMNET"]  # 色彩增强
        self.max_label_length = conf["MAX_LABEL_LENGTH"]
        self.char_dirctory_file = conf["CHAR_DICTORY_FILE"]
        path = conf["TRAIN_DATASET"] if self.train_mode else conf["TEST_DATASET"]  # 数据集路径
        # 获取缓存数据， 如没有则自动创建并保存
        cache = self.get_cache(path, describe, "ocr")
        # 更新
        self.image_list = cache.pop('image')
        self.label_list = cache.pop('label')

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img_label = self.label_list[index]
        # 加载图片
        image = load_image(img_path)
        # 无失真调整到self.imgsz大小
        image, ori_img_shape, scale, fill_w_h_dist = resize_image_tough(image, self.imgsz)
        # 是否训练模式
        if self.train_mode:
            # 色彩增强
            if random.random() < self.color_augment:
                image = ColorAugment(image)
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
        # 标签统一填充到最大字符长度
        label_length = len(img_label)
        for _ in range(label_length, self.max_label_length):
            img_label.append(-1)
        item = {"image": torch.from_numpy(image),
                "label": torch.from_numpy(np.array(img_label, dtype=np.float32)),
                "label_length": label_length}
        # torch.from_numpy(np.array(label_length, dtype=np.float32))
        return item

    def __len__(self):
        return len(self.image_list)