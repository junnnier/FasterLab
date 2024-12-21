import os


def main(root_dir, save_dir):
    result = []
    rain_path = os.path.join(root_dir, "rain")
    norain_path = os.path.join(root_dir, "norain")
    rain_image_list = os.listdir(rain_path)
    for i, image_name in enumerate(rain_image_list):
        rain_image_absoult_path = os.path.join(rain_path, image_name)
        norain_image_absoult_path = os.path.join(norain_path, "no" + image_name)
        if os.path.exists(norain_image_absoult_path):
            result.append("{}\t{}".format(rain_image_absoult_path, norain_image_absoult_path))
    with open(save_dir, "w", encoding="utf-8") as f:
        for item in result:
            f.write(item + "\n")
    print("complete directory:{}\nnumber:{}\tsuccess:{}".format(root_dir, len(rain_image_list), len(result)))


if __name__ == '__main__':
    root_dir="/mnt/work01/project/wantianjun/general_detection/other_dataset/Rain100H_old_version/RainTestH/small"
    save_dir="/mnt/work01/project/wantianjun/general_detection/other_dataset/Rain100H_old_version/test.txt"
    main(root_dir, save_dir)
    print("end")