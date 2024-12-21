import os


def switch_label(current_path):
    result=[]
    image_list = os.listdir(current_path)
    for index, image_name in enumerate(image_list):
        class_name = image_name.split(".")[0].split("_",maxsplit=1)[1]
        result.append(os.path.join(current_path, image_name)+ "\t"+ class_name)
    return result


def main(root_dir):
    dir_list=os.listdir(root_dir)
    for i, dir in enumerate(dir_list):
        label_info=switch_label(os.path.join(root_dir,dir))
        save_label_path=os.path.join(root_dir, dir + ".txt")
        with open(save_label_path,"w",encoding="utf-8") as f:
            for item in label_info:
                f.write(item + "\n")
        print("complete directory:{}\tnumber:{}".format(dir,len(label_info)))


if __name__ == '__main__':
    root_dir="D:\\dataset\\cifar100"
    main(root_dir)
    print("end")