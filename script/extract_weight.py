import torch
import yaml
import argparse
import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from utils.select_tools import get_network


def main(args):
    # 读取配置文件
    with open(args.config, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    net = get_network(conf).to("cpu")
    # 加载模型权重、网络权重
    ckpt_weight = torch.load(args.weight, map_location="cpu")
    net_weight = net.state_dict().copy()
    net_name_list = list(net_weight.keys())
    ckpt_name_list = list(ckpt_weight.keys())
    net_layer_num = len(net_name_list)
    ckpt_layer_num = len(ckpt_name_list)
    print("net layer:{}\tweight layer:{}".format(net_layer_num, ckpt_layer_num))

    print("-----------------------------")
    # 层数相等替换
    if net_layer_num == ckpt_layer_num:
        new_weight = {}
        for index in range(net_layer_num):
            net_layer = net_name_list[index]
            ckpt_layer = ckpt_name_list[index]
            # shape一样，直接替换
            if net_weight[net_layer].shape == ckpt_weight[ckpt_layer].shape:
                new_weight[net_layer] = ckpt_weight[ckpt_layer]
                print("replace {}\t{}\t{}\t{}".format(net_layer, net_weight[net_layer].shape, ckpt_weight[ckpt_layer].shape, ckpt_layer))
        # 把权重加载到模型中
        net.load_state_dict(new_weight, strict=False)
        # 保存当前模型
        ckpt = {"model": net}
        save_path = "{}-fasterlab.pth".format(args.weight.split(".")[0])
        torch.save(ckpt, save_path)
        print("save success to: {}".format(save_path))
    else:
        max_layer_num = max(net_layer_num, ckpt_layer_num)
        if net_layer_num < max_layer_num:
            net_name_list.extend(["None"]*(max_layer_num - net_layer_num))
        else:
            ckpt_name_list.extend(["None"]*(max_layer_num - ckpt_layer_num))
        for index in range(max_layer_num):
            print("{}\t{}".format(net_name_list[index], ckpt_name_list[index]))


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="../config/cifar100.yaml", help='experiment config file')
    parser.add_argument('--weight', type=str, default="RepVGG-A0-train.pth", help='Pre-training weight')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_opt()
    main(args)
    print("end")
