import os
import hashlib
from PIL import Image
import numpy as np


def check_image(img_path):
    """
    Args:
        img_path: 图片路径
    Returns: True/False
    """
    if os.path.exists(img_path):
        try:
            im = Image.open(img_path)
            im.verify()
            return img_path
        except:
            return False
    else:
        return False


def check_classify_label(label, label_index_name):
    """
    Args:
        label: 所有图片的标签名称的列表
        label_index_name: 标签名称列表索引
    Returns: list or False
    """
    result = []
    right_signal = True
    for item in label:
        try:
            result.append(label_index_name.index(item))
        except:
            right_signal = False
            break
    if right_signal:
        return result
    else:
        return False


def check_detect_label(label_path):
    """
    Args:
        label_path: 标签文件路径
    Returns: numpy data or False
    """
    if os.path.isfile(label_path):
        with open(label_path, 'r') as f:
            label_data = [x.split() for x in f.read().strip().splitlines() if len(x)]
            label_data = np.array(label_data, dtype=np.float32)
        if len(label_data):
            assert label_data.shape[1] == 5, "Labels require 5 columns each. {}".format(label_path)
            assert (label_data >= 0).all(), "Wrong value, the value must be greater than 0. {}".format(label_path)
            assert (label_data[:, 1:] <= 1).all(), 'The label are not normalized or boxes coordinate are out of bounds. {}'.format(label_path)
            assert np.unique(label_data, axis=0).shape[0] == label_data.shape[0], "Duplicate labels. {}".format(label_path)
        else:
            label_data = np.zeros((0, 5), dtype=np.float32)
        return label_data
    else:
        return False


def get_hash(path):
    """
    Args:
        path: 文件路径
    Returns: md5值
    """
    with open(path, 'rb') as f:
        file = f.read()
    md5 = hashlib.md5(file).hexdigest()
    return md5