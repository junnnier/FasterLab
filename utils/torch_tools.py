import time
import torch
import torch.nn as nn
from contextlib import contextmanager
import torch.distributed as dist


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)  # 模型是否为DP或者DDP模式


def de_parallel(model):
    return model.module if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel) else model


def replace_weight(ckpt_weight,model_weight,exclude=()):
    return {k: v for k, v in ckpt_weight.items() if k in model_weight and not any(x in k for x in exclude) and v.shape == model_weight[k].shape}


@contextmanager
def torch_distributed_zero_first(local_rank):
    # 进程不是主进程，通过barrier()设置一个阻塞栅栏，让此进程处于等待状态。是主进程通过yield跳转回去，运行with作用域范围内的代码。
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank]) if dist.is_nccl_available() else dist.barrier()
    yield
    # 主进程执行完代码再次回到这里，此时其它进程都到达了当前的栅栏处，通过barrier()释放栅栏，这样所有进程就达到了同步。
    if local_rank == 0:
        dist.barrier(device_ids=[0]) if dist.is_nccl_available() else dist.barrier()


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()