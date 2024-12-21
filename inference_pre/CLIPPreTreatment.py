import torch
import cv2
from transformers import DistilBertTokenizer
import albumentations as A
# --------本地导入--------
from utils.image_tools import load_image


class PreTreatment(object):
    def __init__(self):
        # 类别名称
        self.evluate_labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        # 生成prompt
        caption_list = ["a photo of {}".format(c) for c in self.evluate_labels]
        # 文本编码器
        tokenizer = DistilBertTokenizer.from_pretrained("/mnt/work01/project/wantianjun/general_detection/other_dataset/Flickr30K/textencoder_model")
        self.encoded_captions = tokenizer(caption_list, padding='max_length', truncation=True, max_length=100, return_tensors='pt')
        # 归一化增强操作
        self.transforms = A.Compose([A.Resize(224, 224, always_apply=True),
                                    A.Normalize(max_pixel_value=255.0, always_apply=True)])

    def __call__(self, image_path, image_size, force_resize, device):
        # 读取图片
        image = load_image(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_shapes = (image.shape, 1, (0,0))
        image = self.transforms(image=image)['image']
        images = {key: torch.tensor(values) for key, values in self.encoded_captions.items()}
        images["image"] = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)
        item = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in images.items()}
        return item, image_shapes