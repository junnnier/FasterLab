import random
import cv2
import imutils
import numpy as np
import torch
# --------本地导入--------
from dataset.Base import BaseDataset
from utils.image_tools import load_image,resize_image,ColorAugment,MixUp


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
        self.label_index_name = conf["LABEL_INDEX_NAME"]  # 标签索引名
        self.color_augment = conf["COLOR_AUGMNET"]  # 色彩增强
        self.flip_td = conf["FLIP_TD"]  # 上下翻转
        self.flip_lr = conf["FLIP_LR"]  # 左右翻转
        self.rotation = [] if conf["ROTATION"] is None else conf["ROTATION"]  # 旋转角度，如[-30,30]
        self.mixup = conf["MixUp"]
        self.label_smooth = conf["LABEL_SMOOTH"]  # 标签平滑epsilon值
        path = conf["TRAIN_DATASET"] if self.train_mode else conf["TEST_DATASET"]  # 数据集路径
        # 获取缓存数据， 如没有则自动创建并保存
        cache = self.get_cache(path, describe, "classify", [self.label_index_name])
        # 更新
        self.image_list = cache.pop('image')
        self.label_list = cache.pop('label')

    def __getitem__(self, index):
        img_path=self.image_list[index]
        img_label=self.label_list[index]
        # 加载图片
        image = load_image(img_path)
        # 无失真调整到self.imgsz大小
        image, ori_img_shape, scale, fill_w_h_dist = resize_image(image,self.imgsz[0])
        image_shapes = (ori_img_shape, scale, fill_w_h_dist)
        # 是否训练模式
        if self.train_mode:
            # 色彩增强
            if random.random() < self.color_augment:
                image = ColorAugment(image)
            # 左右翻转
            if random.random() < self.flip_lr:
                image = cv2.flip(image,1)
            # 上下翻转
            if random.random() < self.flip_td:
                image = cv2.flip(image,0)
            # 混合图片
            if random.random() < self.mixup:
                new_index = random.randint(0, len(self.image_list) - 1)
                img2_path = self.image_list[new_index]
                img2_label = self.label_list[new_index].copy()
                image2, ori_img_shape, scale, fill_w_h_dist = resize_image(load_image(img2_path), self.imgsz[0])
                image = MixUp(image, image2)
                img_label.extend(img2_label)
            # 旋转
            if self.rotation and random.random() < 0.5:
                angle = random.randint(self.rotation[0], self.rotation[1])
                # 固定size旋转
                image = imutils.rotate(image,angle)
                # 不剪切原图旋转
                # image = imutils.rotate_bound(image,angle)
                # image, _, _, _ = resize_image(image,self.imgsz[0], fill_color=(0, 0, 0))
        # 转换
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image,dtype=np.float32)  # 转换为内存连续存储的数组
        image = image / 255.0
        # 标签one hot编码
        one_hot = self.one_hot_encode(img_label,len(self.label_index_name))
        # 标签平滑
        if self.train_mode and self.label_smooth:
            one_hot = self.label_smoothing(one_hot, len(self.label_index_name), self.label_smooth)
        return torch.from_numpy(image), torch.from_numpy(one_hot), image_shapes

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        img, label, shape = zip(*batch)
        return torch.stack(img, 0), torch.stack(label,0), shape

    def one_hot_encode(self, img_label, class_num):
        encode = np.eye(class_num,dtype=np.float32)[img_label]
        encode = encode.sum(axis=0)
        return encode

    def label_smoothing(self, one_hot, class_num, epsilon=0.1):
        one_hot = one_hot * (1 - epsilon)
        one_hot[np.where(one_hot == 0)] = epsilon / (class_num - 1)
        return one_hot