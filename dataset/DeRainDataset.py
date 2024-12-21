import random
import cv2
import imutils
import numpy as np
import torch
# --------本地导入--------
from dataset.Base import BaseDataset
from utils.image_tools import load_image,RandomCrop,RainAugment


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
        self.flip_td = conf["FLIP_TD"]  # 上下翻转
        self.flip_lr = conf["FLIP_LR"]  # 左右翻转
        self.rotation = [] if conf["ROTATION"] is None else conf["ROTATION"]  # 旋转角度，如[-30,30]
        self.random_crop = conf["RANDOM_CROP"]
        self.rain_augment = conf["RAIN_AUGMENT"]  # 雨量增强
        path = conf["TRAIN_DATASET"] if self.train_mode else conf["TEST_DATASET"]  # 数据集路径
        # 获取缓存数据， 如没有则自动创建并保存
        cache = self.get_cache(path, describe, "segmentation")
        # 更新
        self.image_list = cache.pop('image')
        self.label_list = cache.pop('label')

    def __getitem__(self, index):
        img_path=self.image_list[index]
        label_path=self.label_list[index]
        # 加载图片
        image_train = load_image(img_path)
        image_gt = load_image(label_path)
        image_train = cv2.cvtColor(image_train, cv2.COLOR_BGR2RGB)
        image_gt = cv2.cvtColor(image_gt, cv2.COLOR_BGR2RGB)
        ori_image_shapes = image_train.shape
        # 是否训练模式
        if self.train_mode:
            # 图片增强
            if self.rain_augment:
                image_train, image_gt = RainAugment(image_train, image_gt, self.rain_augment)
            # 随机裁剪
            if self.random_crop:
                image_train, image_gt = RandomCrop([image_train, image_gt], self.imgsz)
            # 左右翻转
            if random.random() < self.flip_lr:
                image_train = cv2.flip(image_train,1)
                image_gt = cv2.flip(image_gt,1)
            # 上下翻转
            if random.random() < self.flip_td:
                image_train = cv2.flip(image_train,0)
                image_gt = cv2.flip(image_gt,0)
            # 旋转
            if self.rotation and random.random() < 0.5:
                angle = random.randint(self.rotation[0], self.rotation[1])
                # 固定size旋转
                image_train = imutils.rotate(image_train,angle)
                image_gt = imutils.rotate(image_gt,angle)
        else:
            height, width, channel = ori_image_shapes
            if height % 16 != 0:
                height = ((height // 16) + 1) * 16
            if width % 16 != 0:
                width = ((width // 16) + 1) * 16
            image_train = cv2.resize(image_train, (width, height))
            image_gt = cv2.resize(image_gt, (width, height))

        image_train = image_train / 255.0
        image_gt = image_gt / 255.0
        image_train = np.ascontiguousarray(image_train,dtype=np.float32)  # 转换为内存连续存储的数组
        image_gt = np.ascontiguousarray(image_gt,dtype=np.float32)  # 转换为内存连续存储的数组
        image_train = torch.from_numpy(image_train.transpose(2, 0, 1)).contiguous()
        image_gt = torch.from_numpy(image_gt.transpose(2, 0, 1)).contiguous()
        return image_train, image_gt, ori_image_shapes

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        img, label, shape = zip(*batch)
        return torch.stack(img, 0), torch.stack(label,0), shape