import os
import torch
from torch.utils.data import DataLoader
from utils.torch_tools import torch_distributed_zero_first
from utils.visual import draw_label_statistics


def get_training_dataloader(conf, dataset_func, word_size, local_rank=-1):
    """
    Args:
        conf: 配置文件内容
        dataset_func: 数据集加载函数
        word_size: 全局中rank的数量
        local_rank: 当前进程的rank号，判断是否DDP模式训练
    Returns: 数据加载器
    """
    # 确保DDP模式下，每个节点只有一个进程来处理加载数据，其他进程可以使用缓存
    with torch_distributed_zero_first(local_rank):
        train_dataset = dataset_func(conf, describe="train")
    sampler = None if local_rank == -1 else torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=None if sampler else True,
                                  batch_size=conf["TRAIN_BATCH_SIZE"] // word_size,
                                  num_workers=conf["NUM_WORKERS"],
                                  sampler=sampler,
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=dataset_func.collate_fn if hasattr(train_dataset, "collate_fn") else None)
    # 绘制标签统计结果
    if local_rank in [-1, 0] and conf.get("LABEL_INDEX_NAME", None):
        draw_label_statistics(train_dataset.label_list,
                              train_dataset.label_index_name,
                              save_path=os.path.join(conf["EXPERIMENT_PATH"], "train_label_statistic.png"))
    return train_dataloader


def get_test_dataloader(conf, dataset_func, word_size):
    """
    Args:
        conf: 配置文件内容
        dataset_func: 数据集加载函数
        word_size: 全局中rank的数量
    Returns: 数据加载器
    """
    test_dataset = dataset_func(conf, describe="test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=conf["TEST_BATCH_SIZE"] // word_size,
                                 num_workers=conf["NUM_WORKERS"],
                                 pin_memory=True,
                                 drop_last=False,
                                 collate_fn=dataset_func.collate_fn if hasattr(test_dataset, "collate_fn") else None)
    # 绘制标签统计结果
    if conf.get("LABEL_INDEX_NAME", None):
        draw_label_statistics(test_dataset.label_list,
                              test_dataset.label_index_name,
                              save_path=os.path.join(conf["EXPERIMENT_PATH"], "test_label_statistic.png"))
    return test_dataloader
