import torch
import numpy as np
import cv2


def path_img2label(img_paths):
    """
    图片路径转换成对应的标签文件路径
    Args:
        img_paths: 图片路径
    Returns: 标签路径
    """
    return img_paths.replace("xiaohongshu_detection_chai","xiaohongshu_detection_label").rsplit('.', 1)[0] + ".txt"


def xyxy2xywh(x):
    # 转换 nx4 boxes 从[x1, y1, x2, y2] to [x, y, w, h]，其中xy1=左上，xy2=右下
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # 将Nx4的box形式从[x, y, w, h]转换为[x1, y1, x2, y2]，其中x1y1为左上角，x2y2为右下角
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nxywh2xyxy(x, w, h, padw=0, padh=0):
    """
    将Nx4的box形式从归一化的[x, y, w, h]转换为[x1, y1, x2, y2]，其中x1y1为左上角，x2y2为右下角
    Args:
        x: Nx4维的归一化坐标
        w: 图片的width
        h: 图片的height
        padw: width偏移量
        padh: height偏移量
    Returns: 转换后的坐标
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2nxywh(x, w, h, clip=False, eps=0.0):
    """
    转换Nx4的box从[x1, y1, x2, y2]转换为归一化的[x, y, w, h]，其中xy1=左上，xy2=右下
    Args:
        x: Nx4维的坐标
        w: 图片的width
        h: 图片的height
        clip: 是否进行坐标裁剪避免超出图片范围
        eps:
    Returns: 转换后的归一化坐标

    """
    if clip:
        clip_coords(x, (h - eps, w - eps))  # 在原图上进行裁剪
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def clip_coords(boxes, shape):
    # 裁剪xyxy边界框到图像shape(height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def coordinate_scale(coord, scale, padw=0, padh=0):
    coord *= scale
    coord[:, 0] += padw
    coord[:, 1] += padh
    coord[:, 2] += padw
    coord[:, 3] += padh


def coordinate_flip_lr(coord, img_w):
    """
    左右翻转Nx4维[x1, y1, x2, y2]格式的box坐标
    Args:
        coord: 坐标
        img_w: 翻转图片的width
    Returns: 翻转后的坐标
    """
    coord[:, 0] = img_w - coord[:, 0]
    coord[:, 2] = img_w - coord[:, 2]


def coordinate_flip_td(coord, img_h):
    """
    上下翻转Nx4维[x1, y1, x2, y2]格式的box坐标
    Args:
        coord: 坐标
        img_h: 翻转图片的height
    Returns: 翻转后的坐标
    """
    coord[:, 1] = img_h - coord[:, 1]
    coord[:, 3] = img_h - coord[:, 3]


def coordinate_rotate_2d(coord, img_w, img_h, rotate_angle=0):
    """
    对Nx2维[x, y]格式的box坐标进行旋转
    Args:
        coord: 坐标
        img_w: 旋转图片的width
        img_h: 旋转图片的height
        rotate_angle: +/-旋转角度
    Returns: 旋转后的坐标
    """
    # 生成旋转矩阵M
    M = cv2.getRotationMatrix2D(center=(img_w/2, img_h/2), angle=rotate_angle, scale=1)
    # 为保留旋转后的完整图片，增加旋转中心点到新图中心点的偏移
    r = np.deg2rad(rotate_angle)
    new_im_width = abs(np.sin(r) * img_h) + abs(np.cos(r) * img_w)
    new_im_height = abs(np.sin(r) * img_w) + abs(np.cos(r) * img_h)
    tx = (new_im_width - img_w) / 2
    ty = (new_im_height - img_h) / 2
    M[0, 2] += tx
    M[1, 2] += ty
    # 生成转置矩阵
    coord_transpose = coord.transpose()
    coord_padded = np.ones((3, coord_transpose.shape[1]), dtype=np.float32)
    coord_padded[0:2, :] = coord_transpose
    # 矩阵相乘
    rotated_coord = np.dot(M, coord_padded)
    return rotated_coord.transpose()