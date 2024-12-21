import os
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from utils.torch_tools import time_sync
from torchvision import transforms as T


# 生成N种三通道的颜色
def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def draw_overlay(image, over_image, save_path):
    fig = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.imshow(over_image, alpha=0.7)
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


class PostProcess(object):
    def __init__(self):
        self.transform = T.Compose([T.Resize(513),
                                    T.CenterCrop(513)])

    def __call__(self, output, shapes, config, image_path, save):
        st = time_sync()
        output = output.max(1)[1].cpu().numpy()[0]  # HW
        et = time_sync() - st
        # 保存
        if save:
            # 预测图
            colorized_preds = voc_cmap()[output].astype('uint8')  # 根据预测的单通道值提取出对应的三通道的颜色
            colorized_preds = Image.fromarray(colorized_preds)
            colorized_preds.save(os.path.join(save, os.path.basename(image_path).split(".")[0]+"_predict.png"))
            # 覆盖图
            image = Image.open(image_path)
            image = self.transform(image)
            draw_overlay(image, output, os.path.join(save, os.path.basename(image_path).split(".")[0]+"_overlay.png"))
        # 打印
        result_info = "{}\tsuccess".format(image_path)
        print(result_info)
        return result_info, et