import os
import numpy as np
import logging
import cv2
cv2.setNumThreads(0)  # 设置cv2单线程防止OpenCV多线程化，降低DataLoader运行时cpu占用率
from torch.utils.data import Dataset
from tqdm import tqdm
# --------本地导入--------
from utils.check_tools import check_image, get_hash, check_classify_label, check_detect_label
from utils.label_tools import path_img2label


class BaseDataset(Dataset):
    def __init__(self):
        self.__data_processing = {
            "classify": self._classify_operate,
            "detect": self._detect_operate,
            "segmentation": self._segmentation_operate,
            "clip": self._clip_operate,
            "ocr": self._ocr_recognize_operate
        }

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __repr__(self):
        format_string = "Dataset: {}\t#data: {}\n".format(self.__class__.__name__, self.__len__())
        return format_string

    def get_cache(self, path, describe, task="", *args, **kwargs):
        # 缓存存放路径
        cache_path = os.path.splitext(path)[0] + ".cache"
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # 加载缓存
            assert cache['hash'] == get_hash(path)  # hash是否相同
        except:
            cache, exists = self.create_cache(path, cache_path, describe, task, *args, **kwargs), False  # 生成缓存数据标签，并保存
        # 获取缓存并直接打印结果
        right_num, error_num, empty_num = cache.pop('results')
        if exists:
            desc = "{} dataset scan | right {} error {} empty {}".format(describe, right_num, error_num, empty_num)
            tqdm(None, desc=desc, total=(right_num + error_num + empty_num), initial=(right_num + error_num + empty_num))  # 显示缓存结果
        return cache

    def create_cache(self, path, cache_path, describe, task, *args, **kwargs):
        image_list = []
        label_list = []
        caption_list = []
        x = {}
        right_num, error_num, empty_num = 0, 0, 0
        # 读取文件
        with open(path, 'r', encoding="utf-8") as f:
            data_list = f.readlines()
        # 创建进度条
        data_bar = tqdm(data_list, ascii=" =", unit="image")
        # 对每条数据操作
        for data in data_bar:
            img_path, img_target, img_caption = self.__data_processing[task](data, *args, **kwargs)
            # 检查图片存在且可用、标签正常
            if check_image(img_path) is not False and img_target is not False:
                image_list.append(img_path)
                label_list.append(img_target)
                if img_caption:
                    caption_list.append(img_caption)
                # 统计数量
                if len(img_target):
                    right_num += 1
                else:
                    empty_num += 1
            else:
                error_num += 1
                logging.info("image or label error: {}".format(img_path))
            data_bar.set_description("{} dataset scan | right {} error {} empty {}".format(describe, right_num, error_num, empty_num))
        data_bar.close()

        x['hash'] = get_hash(path)
        x['results'] = right_num, error_num, empty_num
        x['image'] = image_list
        x['label'] = label_list
        if caption_list:
            x['caption'] = caption_list
        try:
            np.save(cache_path, x)  # 保存缓存以备下次使用
            os.rename(cache_path + ".npy", cache_path)  # 去除.npy后缀
            logging.info("New cache created: {}".format(cache_path))
        except Exception as e:
            logging.info(" {} is not writeable: {}".format(cache_path, e))
        return x

    def _classify_operate(self, data, *args, **kwargs):
        img_path, img_label = data.strip().split()
        img_target = check_classify_label(img_label.split(","), *args[0])
        img_caption = None
        return img_path, img_target, img_caption

    def _detect_operate(self, data, *args, **kwargs):
        img_path = data.strip()
        img_label = path_img2label(img_path)
        img_target = check_detect_label(img_label)
        img_caption = None
        return img_path, img_target, img_caption

    def _segmentation_operate(self, data, *args, **kwargs):
        img_path, img_label = data.strip().split()
        img_target = check_image(img_label)
        img_caption = None
        return img_path, img_target, img_caption

    def _clip_operate(self, data, *args, **kwargs):
        img_path, img_caption, img_target = data.strip().split("\t")
        return img_path, img_target, img_caption

    def _ocr_recognize_operate(self, data, *args, **kwargs):
        img_path, img_label = data.strip().split("\t")
        img_target = list(map(int, img_label.split(" ")))
        img_caption = None
        return img_path, img_target, img_caption
