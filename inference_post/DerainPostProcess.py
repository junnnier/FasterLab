import numpy as np
import cv2
import os
from utils.torch_tools import time_sync


class PostProcess(object):
    def __init__(self):
        pass

    def __call__(self, output, shapes, config, image_path, save):
        st = time_sync()
        ori_image_height, ori_iamge_width, ori_image_channel = shapes[0]
        # 计算填充的大小
        output_image = output * 255.0
        output_image = output_image.data.permute(0, 2, 3, 1).cpu().numpy()
        output_image = np.clip(output_image, 0, 255)
        output_image = output_image.astype(np.uint8)[0, :, :, :]
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        output_image = cv2.resize(output_image, (ori_iamge_width, ori_image_height))
        et = time_sync() - st
        # 保存预测结果图片
        if save:
            save_img_name = os.path.splitext(os.path.basename(image_path))[0] + "_predict.png"
            save_img_path = os.path.join(save, save_img_name)
            cv2.imwrite(save_img_path, output_image)
        # 打印
        result_info = "{}\tsuccess".format(image_path)
        print(result_info)
        return result_info, et
