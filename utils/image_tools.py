import os
import cv2
import random
import numpy as np
# --------本地导入--------
from extension.DerainAugmentMix import derain_augment_mix
from utils.label_tools import nxywh2xyxy,coordinate_scale


def load_image(path):
    img = cv2.imread(path)  # 图片为BGR
    return img


def resize_image(im, imgsz, fill_color=(114, 114, 114)):
    ori_img_shape = im.shape
    ori_h, ori_w, ori_c = ori_img_shape
    r = imgsz / max(ori_h, ori_w)  # 计算最大边要调整到输入图片大小的比例，无失真缩放
    # 比例不相等进行缩放
    if r != 1:
        im = cv2.resize(im, (int(ori_w * r), int(ori_h * r)), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    # 填充
    dw, dh = 0, 0
    if fill_color:
        new_h, new_w, new_c = im.shape
        dw = (imgsz-new_w)/2
        dh = (imgsz-new_h)/2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算填充开始的位置
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)  # 填充
    fill_w_h = (dw, dh)
    return im, ori_img_shape, r, fill_w_h


def resize_image_tough(im, imgsz):
    ori_img_shape = im.shape
    im = cv2.resize(im, (imgsz[0], imgsz[1]))
    return im, ori_img_shape, 0, (0, 0)


def ColorAugment(image, hgain=0.015, sgain=0.6, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))  # 转换到HSV通道
    dtype = image.dtype  # uint8

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    image = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
    return image


def ImageRotate(image,rotation):
    height, width, _ = image.shape
    center = (width / 2, height / 2)  # 绕图片中心进行旋转
    angle = random.randint(rotation[0], rotation[1])
    scale = 1.0  # 图像缩放为原来的多少倍
    M = cv2.getRotationMatrix2D(center, angle, scale)  # 获得旋转矩阵
    image_rotation = cv2.warpAffine(src=image, M=M, dsize=(height, width), borderValue=(114, 114, 114))  # 进行仿射变换，默认填充黑色
    return image_rotation


def MixUp(image1, image2):
    r = np.random.beta(32.0, 32.0)  # 混合比例
    image = (image1 * r + image2 * (1 - r)).astype(np.uint8)
    return image


def RandomCrop(image_list, crop_size):
    result_list = []
    ih, iw = image_list[0].shape[:2]
    h1 = random.randint(0, ih-min(ih, crop_size[0]))
    w1 = random.randint(0, iw-min(iw, crop_size[1]))
    h2 = h1 + crop_size[0]
    w2 = w1 + crop_size[1]
    for img in image_list:
        if len(img.shape) == 3:
            result_list.append(img[h1:h2, w1:w2, :])
        else:
            result_list.append(img[h1:h2, w1:w2])
    return result_list[0] if len(result_list) == 1 else result_list


def RainAugment(image_train, image_gt, root_path):

    def getRandRainLayer2(root_path):
        # 随机生成id值
        rand_id1 = random.randint(1, 165)
        rand_id2 = random.randint(4, 8)
        # 获取一张雨量样本图
        rainlayer_image_path = os.path.join(root_path, str(rand_id1) + "-" + str(rand_id2) + ".png")
        rainlayer_rand = load_image(rainlayer_image_path).astype(np.float32) / 255.0
        rainlayer_rand = cv2.cvtColor(rainlayer_rand, cv2.COLOR_BGR2RGB)
        return rainlayer_rand

    image_train = (image_train.astype(np.float32)) / 255.0
    image_gt = (image_gt.astype(np.float32)) / 255.0

    img_rainy_ret = image_train if random.randint(0, 10) > 3 else image_gt
    img_gt_ret = image_gt

    # 随机获取雨层并增强
    rainlayer_rand2 = getRandRainLayer2(root_path)
    rainlayer_aug2 = derain_augment_mix(rainlayer_rand2, severity=3, width=3, depth=-1) * 1

    # 对雨层裁剪
    height = min(image_gt.shape[0], rainlayer_aug2.shape[0])
    width = min(image_gt.shape[1], rainlayer_aug2.shape[1])
    rainlayer_aug2_crop = RandomCrop([rainlayer_aug2], (height, width))

    img_gt_ret, img_rainy_ret = RandomCrop([img_gt_ret, img_rainy_ret], (height, width))
    img_rainy_ret = img_rainy_ret + rainlayer_aug2_crop - img_rainy_ret * rainlayer_aug2_crop
    np.clip(img_rainy_ret, 0.0, 1.0)

    img_rainy_ret = img_rainy_ret * 255
    img_gt_ret = img_gt_ret * 255

    return img_rainy_ret, img_gt_ret


def MosaicAugment(self, index):
    labels4 = []
    image_size = self.imgsz[0]
    # 把image_size扩大一倍，然后在一定范围内，随机生成大图的中心点x, y
    yc, xc = [int(random.uniform(image_size//2, (2*image_size)-(image_size//2))) for i in range(2)]
    # 随机添加3个附加的图片索引
    indices = [index] + random.choices(range(len(self.image_list)), k=3)
    random.shuffle(indices)
    # 对每张图片操作
    for i, indice in enumerate(indices):
        image = load_image(self.image_list[indice])
        # 无失真调整到self.imgsz大小，不进行填充
        image, ori_img_h_w_c, scale, w_h_fill_dist = resize_image(image, image_size, fill_color=None)
        img_h, img_w, img_c = image.shape
        # 将img放在img4这个大图中
        if i == 0:  # top left
            img4 = np.full((image_size * 2, image_size * 2, ori_img_h_w_c[2]), 114, dtype=np.uint8)  # 初始化大图img4
            x1a, y1a, x2a, y2a = max(xc - img_w, 0), max(yc - img_h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image) 设置在大图上的位置，左上角和右下角
            x1b, y1b, x2b, y2b = img_w - (x2a - x1a), img_h - (y2a - y1a), img_w, img_h  # xmin, ymin, xmax, ymax (small image) 在小图上的位置
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - img_h, 0), min(xc + img_w, image_size * 2), yc
            x1b, y1b, x2b, y2b = 0, img_h - (y2a - y1a), min(img_w, x2a - x1a), img_h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - img_w, 0), yc, xc, min(image_size * 2, yc + img_h)
            x1b, y1b, x2b, y2b = img_w - (x2a - x1a), 0, img_w, min(y2a - y1a, img_h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + img_w, image_size * 2), min(image_size * 2, yc + img_h)
            x1b, y1b, x2b, y2b = 0, 0, min(img_w, x2a - x1a), min(y2a - y1a, img_h)
        # 将小图上截取的部分贴到大图img4上[ymin:ymax, xmin:xmax]
        img4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

        # 计算小图到大图上产生的偏移
        padw = x1a - x1b
        padh = y1a - y1b
        # 调整标签，转换为马赛克图片中xyxy坐标位置
        labels = self.label_list[indice].copy()
        if labels.size:
            labels[:, 1:5] = nxywh2xyxy(labels[:, 1:5], img_w, img_h, padw=padw, padh=padh)
        labels4.append(labels)

    # 目标框坐标可能在大图的外面，所以进行裁剪标签
    labels4 = np.concatenate(labels4, 0)  # 把4个小图上的标签拼接起来
    np.clip(labels4[:, 1:5], 0, 2*image_size, out=labels4[:, 1:5])  # 裁剪
    # 图片和目标框缩放到image_size大小
    img4, _, r, _ = resize_image(img4, image_size, fill_color=(114,114,114))
    coordinate_scale(labels4[:, 1:5], 0.5)

    return img4, labels4
