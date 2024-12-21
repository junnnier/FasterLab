import os


def main(root_dir, data_file):
    with open(os.path.join(root_dir, data_file), "r") as f:
        data_list = f.readlines()

    result = []
    for data in data_list:
        data = data.strip()
        # 图片路径
        image_dir = os.path.join(root_dir, "JPEGImages")
        image_path = os.path.join(image_dir, data + ".jpg")
        # 分割图路径
        iamge_segm_dir = os.path.join(root_dir, "SegmentationClass")
        image_segm_path = os.path.join(iamge_segm_dir, data + ".png")
        # 记录
        result.append("\t".join([image_path, image_segm_path]))

    save_path = os.path.join(root_dir, os.path.basename(data_file))
    with open(save_path,"w",encoding="utf-8") as f:
        for item in result:
            f.write(item + "\n")
    print("number {}\nsave to {}".format(len(result), save_path))


if __name__ == '__main__':
    root_dir = "/mnt/work01/project/wantianjun/general_detection/other_dataset/VOCdevkit/VOC2012/"
    data_file = "ImageSets/Segmentation/val.txt"
    main(root_dir, data_file)
    print("end")