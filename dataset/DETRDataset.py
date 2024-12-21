import torch
import torchvision
from pycocotools import mask as coco_mask
# --------本地导入--------
from extension.detr_module import dataset_augment as detr_T
from extension.detr_module.tools import nested_tensor_from_tensor_list


class DatasetTask(torchvision.datasets.CocoDetection):
    def __init__(self, conf, describe=""):
        """
        Args:
            conf: 配置文件
            describe: train还是test
        """
        self.train_mode = True if describe == "train" else False
        img_folder, ann_file = conf["TRAIN_DATASET"] if self.train_mode else conf["TEST_DATASET"]  # 数据集路径
        super(DatasetTask, self).__init__(img_folder, ann_file)
        self.imgsz = conf["IMAGE_SIZE"]  # 图片尺寸
        self.mean = conf["NORMALIZE_MEAN"]
        self.std = conf["NORMALIZE_STD"]
        self.return_masks = conf["RETURN_MASK"]
        self.train_data_augment = self.train_augment()
        self.test_data_augment = self.test_augment()

    def __getitem__(self, index):
        img, target = super(DatasetTask, self).__getitem__(index)
        image_id = self.ids[index]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.conver_coco_format(img, target)
        if self.train_mode:
            img, target = self.train_data_augment(img, target)
        else:
            img, target = self.test_data_augment(img, target)
        return img, target

    def train_augment(self):
        normalize = detr_T.Compose([
            detr_T.ToTensor(),
            detr_T.Normalize(self.mean, self.std)
        ])
        return detr_T.Compose([
            detr_T.RandomHorizontalFlip(),
            detr_T.RandomSelect(
                detr_T.RandomResize(self.imgsz, max_size=1333),
                detr_T.Compose([
                    detr_T.RandomResize([400, 500, 600]),
                    detr_T.RandomSizeCrop(384, 600),
                    detr_T.RandomResize(self.imgsz, max_size=1333),
                ])
            ),
            normalize,
        ])

    def test_augment(self):
        normalize = detr_T.Compose([
            detr_T.ToTensor(),
            detr_T.Normalize(self.mean, self.std)
        ])
        return detr_T.Compose([
            detr_T.RandomResize([800], max_size=1333),
            normalize,
        ])

    def convert_coco_poly_to_mask(self, segmentations, height, width):
        masks = []
        for polygons in segmentations:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
            masks.append(mask)
        if masks:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        return masks

    def conver_coco_format(self, image, target):
        w, h = image.size
        # 获取图片id
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        # 获取注释，去除拥挤的图片
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = [obj["bbox"] for obj in anno]
        # 调整boxes
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        # 获取类别
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        # 是否返回分割的mask
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = self.convert_coco_poly_to_mask(segmentations, h, w)
        # 获取keypoints
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # 转换为coco的api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

    @staticmethod
    def collate_fn(batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0])
        return tuple(batch)