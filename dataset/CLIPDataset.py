import random
import cv2
import torch
import albumentations as A
# --------本地导入--------
from dataset.Base import BaseDataset
from utils.image_tools import load_image
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer


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
        self.caption_list = []
        self.imgsz = conf["IMAGE_SIZE"]  # 图片尺寸
        path = conf["TRAIN_DATASET"] if self.train_mode else conf["TEST_DATASET"]  # 数据集路径
        # 获取缓存数据， 如没有则自动创建并保存
        cache = self.get_cache(path, describe, "clip")
        # 更新
        self.image_list = cache.pop('image')
        self.label_list = cache.pop('label')
        self.caption_list = cache.pop('caption')
        self.init_len = len(self.caption_list)

        # resize和归一化操作
        self.transforms = A.Compose([A.Resize(self.imgsz[0], self.imgsz[1], always_apply=True),
                                    A.Normalize(max_pixel_value=255.0, always_apply=True)])

        # Bert模型的分词器
        self.tokenizer = DistilBertTokenizer.from_pretrained(conf["TEXT_TOKENIZER"])
        # 生成文本初始编码，返回一个字典{'input_ids'：[索引], 'attention_mask'：[文本特征]}
        self.encoded_captions = self.tokenizer(self.caption_list, padding=True, truncation=True, max_length=conf["MAX_LENGTH"])

    def __getitem__(self, index):
        if index >= self.init_len:
            index = index - self.init_len
            img_path = self.image_list[index]
            img_caption = self.caption_list[index]
            # 加载图片
            image = load_image(img_path)
            # 文本编码特征
            item = {key: torch.tensor(values[index]) for key, values in self.encoded_captions.items()}

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transforms(image=image)['image']
            item['image'] = torch.tensor(image).permute(2, 0, 1).float()
            item['caption'] = img_caption

        else:
            img_path = self.image_list[index]
            img_caption = self.caption_list[index]
            # 加载图片
            image = load_image(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transforms(image=image)['image']

            rand_index = random.randint(0, self.init_len - 1)
            image_b = load_image(self.image_list[rand_index])
            image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
            image_b = self.transforms(image=image_b)['image']
            avg_image = (image + image_b) / 2

            # 文本编码特征
            item = {key: torch.tensor(values[index]) for key, values in self.encoded_captions.items()}
            item['image'] = torch.tensor(avg_image).permute(2, 0, 1).float()
            item['caption'] = img_caption + " " + self.caption_list[rand_index]

        return item

    def __len__(self):
        return len(self.image_list) * 2