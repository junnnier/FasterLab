from collections import defaultdict
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from copy import copy


def draw_label_statistics(label_list, label_index_name, save_path="label_statistic.png"):
    """
    Args:
        label_list: 所有图片的标签索引的列表
        label_index_name: 标签索引对应的名称列表
        save_path: 保存图片路径
    """
    statistic_result = defaultdict(int)
    for img_label in label_list:
        if isinstance(img_label, np.ndarray):
            img_label = img_label.copy()[:, 0].astype(np.int8)
        for label in img_label:
            statistic_result[label_index_name[label]] += 1
    x_value = []
    y_value = []
    for key,value in statistic_result.items():
        x_value.append(key)
        y_value.append(value)
    plt.figure()
    rects = plt.bar(x_value, y_value, width=0.8)
    plt.xticks(range(len(x_value)), x_value, rotation=45)
    plt.xlabel("category name")
    plt.ylabel("number")
    plt.title('Category number statistics')
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig(save_path)
    print("category statistic save to: {}".format(save_path))


def derain_image_save(sample_folder, sample_name, img_list, name_list, pixel_max_cnt=255, height=-1, width=-1):
    height = height if isinstance(height,int) else height.item()
    width = width if isinstance(width,int) else width.item()
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        img = img * 255.0  # Recover normalization
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        if (height != -1) and (width != -1):
            img_copy = cv2.resize(img_copy, (width, height))
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)


def draw_box_label(image, box, label="", color=(0, 0, 255), txt_color=(255, 255, 255)):
    image_copy = copy(image)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))  # box的左上角和右下角坐标
    line_width = max(round(sum(image_copy.shape) / 2 * 0.003), 2)  # 计算box的线宽
    # 矩形框
    cv2.rectangle(image_copy, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)
    # 文本
    if label:
        font_thickness = max(line_width - 1, 1)
        w, h = cv2.getTextSize(label, 0, fontScale=line_width / 3, thickness=font_thickness)[0]
        outside = p1[1] - h - 3 >= 0  # 绘制标签是否会超出图片
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image_copy, p1, p2, color, -1, cv2.LINE_AA)  # 填充字体背景
        cv2.putText(image_copy, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, line_width / 3, txt_color, thickness=font_thickness, lineType=cv2.LINE_AA)
    return image_copy


def draw_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close()


def draw_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close()


def save_crop_box(xyxy, image, square=False):
    image_copy = copy(image)
    img = image_copy[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
    if square:
        img_h, img_w, img_c = img.shape
        if img_h > img_w:
            dw = (img_h - img_w) / 2
            dh = 0
        else:
            dw = 0
            dh = (img_w - img_h) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算填充开始的位置
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 填充
    return img