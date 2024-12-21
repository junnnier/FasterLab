from PIL import Image
# --------本地导入--------
from dataset.Base import BaseDataset
from extension.deeplabv3_module import dataset_augment as et


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
        self.mean = conf["NORMALIZE_MEAN"]
        self.std = conf["NORMALIZE_STD"]
        path = conf["TRAIN_DATASET"] if self.train_mode else conf["TEST_DATASET"]  # 数据集路径
        # 获取缓存数据， 如没有则自动创建并保存
        cache = self.get_cache(path, describe, "segmentation")
        # 更新
        self.image_list = cache.pop('image')
        self.label_list = cache.pop('label')
        assert len(self.image_list) == len(self.label_list), "The number of trained images and labels are not equal."
        self.train_data_augment = self.train_augment()
        self.test_data_augment = self.test_augment()

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label_path = self.label_list[index]
        # 加载图片
        image = Image.open(img_path).convert('RGB')
        segment_img = Image.open(label_path)
        # 是否训练模式
        if self.train_mode:
            img, segment_img = self.train_data_augment(image, segment_img)
        else:
            img, segment_img = self.test_data_augment(image, segment_img)
        return img, segment_img

    def train_augment(self):
        train_transform = et.ExtCompose([
            et.ExtResize(size=self.imgsz[0]),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=self.imgsz, pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=self.mean,
                            std=self.std),
        ])
        return train_transform

    def test_augment(self):
        val_transform = et.ExtCompose([
            et.ExtResize(self.imgsz[0]),
            et.ExtCenterCrop(self.imgsz[0]),
            et.ExtToTensor(),
            et.ExtNormalize(mean=self.mean,
                            std=self.std),
        ])
        return val_transform

    def __len__(self):
        return len(self.image_list)