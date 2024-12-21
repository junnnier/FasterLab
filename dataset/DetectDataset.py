import random
import cv2
import numpy as np
import torch
# --------本地导入--------
from dataset.Base import BaseDataset
from utils.image_tools import load_image,resize_image,MixUp,MosaicAugment
from utils.label_tools import nxywh2xyxy,xyxy2nxywh,coordinate_scale


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
        self.label_index_name = conf["LABEL_INDEX_NAME"]  # 标签索引名
        self.imgsz = conf["IMAGE_SIZE"]  # 图片尺寸
        self.color_augment = conf["COLOR_AUGMNET"]  # 色彩增强
        self.mosaic = conf["MOSAIC"]
        self.mixup = conf["MixUp"]
        self.flip_td = conf["FLIP_TD"]  # 上下翻转
        self.flip_lr = conf["FLIP_LR"]  # 左右翻转
        path = conf["TRAIN_DATASET"] if self.train_mode else conf["TEST_DATASET"]  # 数据集路径
        # 获取缓存数据， 如没有则自动创建并保存
        cache = self.get_cache(path, describe, "detect")
        # 更新
        self.image_list = cache.pop('image')
        self.label_list = cache.pop('label')
        assert len(self.image_list) == len(self.label_list), "The number of trained images and labels are not equal."

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img_label = self.label_list[index].copy()
        label_num = len(img_label)
        # 是否训练模式
        if self.train_mode:
            # 是否马赛克增强
            if random.random() < self.mosaic:
                image, img_label = MosaicAugment(self, index)
                # 混合增强
                if random.random() < self.mixup:
                    image2, img2_label = MosaicAugment(self, random.randint(0, len(self.image_list) - 1))
                    image = MixUp(image, image2)
                    img_label = np.concatenate((img_label, img2_label), 0)
                image_shapes = (image.shape, 1, (0, 0))
            else:
                image = load_image(img_path)  # 加载图片
                # 无失真调整到self.imgsz大小
                image, ori_img_shape, scale, fill_w_h_dist = resize_image(image, self.imgsz[0])
                image_shapes = (ori_img_shape, scale, fill_w_h_dist)
                # 如果有目标框，将归一化的xywh坐标调整为xyxy格式，并调整到缩放和填补后的位置
                if label_num:
                    img_label[:, 1:5] = nxywh2xyxy(img_label[:, 1:5], ori_img_shape[1], ori_img_shape[0])
                    coordinate_scale(img_label[:, 1:5], scale, padw=fill_w_h_dist[0], padh=fill_w_h_dist[1])
            # 如果有目标框，将xyxy坐标调整为归一化的xywh格式
            if label_num:
                img_label[:, 1:5] = xyxy2nxywh(img_label[:, 1:5], image.shape[1], image.shape[0], clip=True, eps=1E-3)
            # 左右翻转
            if random.random() < self.flip_lr:
                image = cv2.flip(image,1)
                if label_num:
                    img_label[:, 1] = 1 - img_label[:, 1]
            # 上下翻转
            if random.random() < self.flip_td:
                image = cv2.flip(image,0)
                if label_num:
                    img_label[:, 2] = 1 - img_label[:, 2]
        else:
            image = load_image(img_path)  # 加载图片
            # 无失真调整到self.imgsz大小
            image, ori_img_shape, scale, fill_w_h_dist = resize_image(image, self.imgsz[0])
            image_shapes = (ori_img_shape, scale, fill_w_h_dist)
            # 如果有目标框，将归一化的xywh坐标调整为xyxy格式，并调整到缩放和填补后的位置，再调整回归一化的xywh格式
            if label_num:
                img_label[:, 1:5] = nxywh2xyxy(img_label[:, 1:5], ori_img_shape[1], ori_img_shape[0])
                coordinate_scale(img_label[:, 1:5], scale, padw=fill_w_h_dist[0], padh=fill_w_h_dist[1])
                img_label[:, 1:5] = xyxy2nxywh(img_label[:, 1:5], image.shape[1], image.shape[0], clip=True, eps=1E-3)

        # 增加一列元素，配合collate_fn使用（在第0个元素添加该label在batch中对应图片的索引）
        img_label_new = np.zeros((len(img_label), 6), dtype=np.float32)
        if label_num:
            img_label_new[:, 1:] = img_label

        # 转换
        image = image / 255.0
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image,dtype=np.float32)  # 转换为内存连续存储的数组
        return torch.from_numpy(image), torch.from_numpy(img_label_new), image_shapes

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        img, label, shape = zip(*batch)
        # 在第0个元素添加目标框在batch中对应图片的索引
        for idx, l in enumerate(label):
            l[:, 0] = idx
        return torch.stack(img, 0), torch.cat(label, 0), shape