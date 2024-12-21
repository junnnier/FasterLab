import cv2
import os
import torch
import json
from utils.torch_tools import time_sync
from extension.yolov5_module.tools import non_max_suppression
from utils.label_tools import coordinate_scale, clip_coords
from utils.visual import draw_box_label, save_crop_box


class PostProcess(object):
    def __init__(self):
        self.conf_thres = 0.2  # 置信度阈值
        self.iou_thres = 0.6  # NMS的IoU阈值
        self.single_cls = False
        self.image_w = 512  # 模型输入图片的宽
        self.image_h = 512  # 模型输入图片的高
        self.save_visual = True  # 是否可视化结果
        self.save_crop = True  # 是否裁剪检测到的box
        self.det_thres = 0.0  # 裁剪和可视化box的阈值

    def __call__(self, output, shapes, config, image_path, save):
        label_index_name = config["LABEL_INDEX_NAME"]
        image = cv2.imread(image_path)

        st = time_sync()
        out, train_out = output
        # NMS
        out = non_max_suppression(out, self.conf_thres, self.iou_thres, multi_label=True, agnostic=self.single_cls)
        et = time_sync() - st

        # 保存格式
        result_info = {"names": "", "positions": "", "scores": ""}
        det_names = []
        det_positions = []
        det_scores = []

        # 统计每张图片结果
        pred = out[0]

        # 是否预测到框
        if len(pred) == 0:
            write_result_string = image_path + "\t" + json.dumps(result_info, separators=(",", ":")) + "\n"
        else:
            # 全部视为单一类数据集
            if self.single_cls:
                pred[:, 5] = 0
            # 预测坐标恢复到原图位置上
            pred[:, 0] += -shapes[2][0]
            pred[:, 1] += -shapes[2][1]
            pred[:, 2] += -shapes[2][0]
            pred[:, 3] += -shapes[2][1]
            coordinate_scale(pred[:, :4], 1 / shapes[1])
            clip_coords(pred[:, :4], shapes[0])

            for *xyxy, conf, cls in reversed(pred):
                det_positions.append(",".join([str(int(x)) for x in torch.tensor(xyxy).view(1, 4).tolist()[0]]))
                det_names.append(label_index_name[int(cls.item())])
                det_scores.append("{:.6}".format(conf.item()))
                # 是否裁剪box
                if save and self.save_crop and conf.item() > self.det_thres:
                    corp_image = save_crop_box([x for x in torch.tensor(xyxy).view(1, 4).tolist()[0]], image)
                    corp_image_name = os.path.join("{}_{}_{}.jpg".format(os.path.basename(image_path).split(".")[0], label_index_name[int(cls.item())], "{:.6}".format(conf.item())))
                    cv2.imwrite(os.path.join(save, corp_image_name), corp_image)
            # 整合最终格式
            result_info["names"] = "_".join(det_names)
            result_info["positions"] = "_".join(det_positions)
            result_info["scores"] = "_".join(det_scores)
            result_info["max_score"] = float(max(det_scores))
            write_result_string = image_path + "\t" + json.dumps(result_info, separators=(",", ":")) + "\n"

        # 是否可视化box
        if save and self.save_visual:
            for index, class_name in enumerate(det_names):
                if float(det_scores[index]) > self.det_thres:
                    image = draw_box_label(image, list(map(int, det_positions[index].split(","))), class_name)
            cv2.imwrite(os.path.join(save, "{}_visual.jpg".format(os.path.basename(image_path).split(".")[0])), image)
        # 打印
        print(write_result_string)
        return write_result_string, et