import torch
from models.yolov5n import yolov5_base


def create_model(param):
    class_num = param["category"]
    depth_multiple = 0.67
    width_multiple = 0.75
    anchors = [[10, 13, 16, 30, 33, 23],  # P3/8
               [30, 61, 62, 45, 59, 119],  # P4/16
               [116, 90, 156, 198, 373, 326]]  # P5/32
    return yolov5_base(class_num, depth_multiple, width_multiple, anchors)


if __name__ == '__main__':
    param = {"category": 2}
    net = create_model(param)
    result = net(torch.zeros(1, 3, 512, 512))
    for index, feature in enumerate(result):
        print("{}--{}".format(index, feature.shape))