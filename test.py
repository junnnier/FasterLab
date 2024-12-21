import argparse
import torch
import os
import yaml
import numpy as np
# --------本地导入--------
from utils.select_tools import select_device,get_pretreatment_function,get_postprocess_function
from utils.torch_tools import time_sync


def test(args):
    device = select_device(args.device)
    image_size = args.image_size
    force_resize = args.force_resize
    if not os.path.exists(args.save):
        os.makedirs(args.save)
        print("create directory...{}".format(args.save))

    print("loading model weights...")
    model = torch.load(args.weight, map_location=device)
    model = model['model']
    model.eval()

    print("loading image file...")
    if os.path.splitext(args.image_path)[1] == ".txt":
        with open(args.image_path,"r",encoding="utf-8") as f:
            image_path_list = f.readlines()
    else:
        image_path_list = [args.image_path]

    print("loading pretreatment function...")
    pretreatment = get_pretreatment_function(args.pretreatment)

    print("loading postprocess function...")
    postprocess = get_postprocess_function(args.postprocess)

    print("loading config file...")
    with open(args.config,"r",encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    consume_time = [0.0, 0.0, 0.0]

    # 对每张图片预测
    for image_path in image_path_list:
        image_path = image_path.strip()

        # 图片是否存在
        if os.path.exists(image_path):
            # 预处理
            t1 = time_sync()
            image, image_shapes = pretreatment(image_path, image_size, force_resize, device)
            t2 = time_sync()
            consume_time[0] += t2 - t1

            # 推理
            with torch.no_grad():
                output = model(image)
            consume_time[1] += time_sync() - t2

            # 后处理
            result_info, spend_time = postprocess(output, image_shapes, conf, image_path, save=args.save)
            consume_time[2] += spend_time

            # 保存
            if args.save:
                with open(os.path.join(args.save, "predict_result.txt"), "a", encoding="utf-8") as f:
                    f.write(result_info)
                    f.write("\n")
        else:
            print("image {} is not exists !".format(image_path))

    average_consume_time = (np.array(consume_time)/len(image_path_list)).tolist()
    print("Total image:{}\nAverage\t[PreProcess: {:.4f}\tModel: {:.4f}\tPostProcess: {:.4f}]".format(len(image_path_list), *average_consume_time))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('--config', type=str, required=True, help='The configuration file')
    parser.add_argument('--force-resize', action="store_true", help='Whether to force resize the image.')
    parser.add_argument('--image-size', type=int, nargs="+", default=[224, 224], help='Enter the images size of the network, default=[224, 224]')
    parser.add_argument('--image-path', type=str, default="", help='To predict the path of an image or path of a file that contains multiple image paths.')
    parser.add_argument('--pretreatment', type=str, default="StandardPreTreatment", help='The name of the function that post-processes the output of the model. default="StandardPreTreatment".')
    parser.add_argument('--postprocess', type=str, required=True, help='The name of the function that post-processes the output of the model.')
    parser.add_argument('--device', type=str, default="0", help="'cpu' or gpu id:'0', default='0'")
    parser.add_argument('--save', type=str, default="", help='predict result saving directory, optional.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_opt()
    test(args)
    print("end")
