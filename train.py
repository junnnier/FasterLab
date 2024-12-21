import os
import argparse
import yaml
import torch
from copy import deepcopy
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# --------本地导入--------
from utils.logger import setup_logger
from utils.select_tools import get_dataset_task, get_network, get_loss_function, get_optimizer, get_scheduler, select_device, get_initialize_weights, get_evaluate_function, get_training_function
from utils.torch_tools import de_parallel, replace_weight
from utils.dataloader import get_training_dataloader, get_test_dataloader
from val import evaluate

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # 在一个node上进程的相对序号，local_rank在node之间相互独立。
RANK = int(os.getenv('RANK', -1))  # 在整个分布式任务中进程的编号/序号，每一个进程对应了一个rank的进程。
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # 全局中（整个分布式任务）rank的数量。


def main(args):
    # 是否恢复训练
    if args.resume:
        config_file=os.path.join(os.path.dirname(os.path.dirname(args.resume)),"config.yaml")
        assert os.path.exists(config_file), "{} is not exist".format(config_file)
        args.config=config_file

    # 读取配置文件
    with open(args.config, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    # 获取实验目录，保存模型路径，保存配置文件
    if args.resume:
        experiment_path=conf["EXPERIMENT_PATH"]
    else:
        experiment_path=os.path.join(conf["LOG_DIR"], conf["EXPERIMENT_NAME"]+datetime.strftime(datetime.now(),"_%Y%m%d_%H%M%S"))
        conf["EXPERIMENT_PATH"] = experiment_path
        conf["WORLD_SIZE"] = WORLD_SIZE
        conf["PRE_WEIGHT"] = args.pre_weight
        conf["FREEZE"] = args.freeze
        if LOCAL_RANK in [-1, 0]:
            os.makedirs(experiment_path)
            with open(os.path.join(experiment_path, "config.yaml"), "w", encoding="utf-8") as f:
                yaml.dump(conf, f, sort_keys=False)

    # 创建日志记录器
    logger = setup_logger(log_dir=experiment_path, rank=RANK)

    logger.info(conf)

    # 保存权重路径
    weights_path = os.path.join(experiment_path, "weights")
    if LOCAL_RANK in [-1, 0] and not os.path.exists(weights_path):
        os.mkdir(weights_path)
    best_pth = os.path.join(weights_path, 'best.pth')
    last_pth = os.path.join(weights_path, 'last.pth')

    # 获取训练设备
    device = select_device(str(conf["DEVICE"]), batch_size=conf["TRAIN_BATCH_SIZE"])
    # DDP模式
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert conf["TRAIN_BATCH_SIZE"] % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        torch.cuda.set_device(LOCAL_RANK)  # 设置环境变量，相当于CUDA_VISIBLE_DEVICES
        device = torch.device('cuda', LOCAL_RANK)  # 指定训练参数要拷入的显卡
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")  # 初始化进程组，windows不支持nccl

    # 获取网络
    net = get_network(conf).to(device)
    # 网络初始化
    if conf["INITIALIZE_WEIGHTS"]:
        get_initialize_weights(net, conf)
    if args.resume or args.pre_weight:
        temp_path = args.resume or args.pre_weight
        ckpt = torch.load(temp_path, map_location=device)  # 加载训练权重
        ckpt_weight = ckpt['model'].float().state_dict() if "model" in ckpt.keys() else ckpt
        new_weight = replace_weight(ckpt_weight, net.state_dict(), exclude=[])  # 筛选出可对应替换的权重
        net.load_state_dict(new_weight, strict=False)  # 把权重加载到模型中
        logger.info("loading weight {} replace {}/{} items".format(temp_path,len(new_weight),len(net.state_dict())))
    # 冻结层
    for k, v in net.named_parameters():
        # 默认训练所有层，如果层名在freeze里面就冻结该层
        v.requires_grad = True
        if k in args.freeze:
            logger.info(f'freezing {k}')
            v.requires_grad = False

    # 数据集加载函数
    dataset_task = get_dataset_task(conf)
    # 训练数据加载:
    train_loader = get_training_dataloader(conf, dataset_task, word_size=WORLD_SIZE, local_rank=LOCAL_RANK)
    # 测试数据加载（只在主进程上进行）
    if RANK in [-1, 0]:
        test_loader = get_test_dataloader(conf, dataset_task, word_size=WORLD_SIZE)
    # 优化器
    optimizer = get_optimizer(net, conf)
    # 学习率衰减scheduler
    train_scheduler = get_scheduler(optimizer, conf)
    # 结果解析函数
    eval_function = get_evaluate_function(conf)
    # 训练函数
    training_function = get_training_function(conf)

    # DP模式
    if RANK == -1 and conf["DEVICE"] != "cpu" and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        logger.info("Use DataParallel mode...")
    # SyncBatchNorm，跨卡BN
    if RANK != -1 and args.sync_bn and conf["DEVICE"] != "cpu":
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
        logger.info("Use SyncBatchNorm()")
    # DDP 模式
    if RANK != -1 and conf["DEVICE"] != "cpu":
        net = DDP(net, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        logger.info("Use DistributedDataParallel mode...")

    # 损失函数
    loss_function = get_loss_function(net, conf)
    # 记录网络图（只在主进程上进行）
    writer = SummaryWriter(log_dir=experiment_path) if RANK in [-1, 0] else None

    # 初始化参数
    best_acc = 0.0
    start_epoch = 0
    total_epoch = conf["EPOCH"]
    if args.resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        train_scheduler.load_state_dict(ckpt["scheduler"])
        best_acc = ckpt['best_acc']
        start_epoch = ckpt['epoch'] + 1

    # -------------------每个epoch进行训练-----------------------
    for epoch in range(start_epoch, total_epoch):

        # 训练
        training_function(net=net,
                          train_loader=train_loader,
                          optimizer=optimizer,
                          device=device,
                          epoch=epoch,
                          total_epoch=total_epoch,
                          RANK=RANK,
                          loss_function=loss_function,
                          logger=logger,
                          writer=writer)

        # 通过scheduler调整学习率
        train_scheduler.step()

        # 验证（只在主进程上进行）
        if RANK in [-1, 0]:
            logger.info('Evaluating Network.....')
            acc, avg_loss = evaluate(model=net,
                                     data_loader=test_loader,
                                     loss_func=loss_function,
                                     eval_func=eval_function,
                                     device=device,
                                     config=conf,
                                     save_dir="")
            logger.info('epoch:{}\taccuracy: {:.4f}\taverage loss: {:.4f}'.format(epoch, acc, avg_loss))
            # 记录
            writer.add_scalar('Test/Average_loss', avg_loss, epoch)
            writer.add_scalar('Test/Accuracy', acc, epoch)
            # 最佳精度
            if acc > best_acc:
                best_acc = acc
            ckpt = {"epoch": epoch,
                    "best_acc": best_acc,
                    "model": deepcopy(de_parallel(net)),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": train_scheduler.state_dict()}
            # 保存当前模型
            torch.save(ckpt, last_pth)
            # 保存最佳模型
            if best_acc == acc:
                torch.save(ckpt, best_pth)
                logger.info('saving best weights file to {}'.format(best_pth))
            # 保存训练中模型
            if conf["INTERVAL_SAVE"] and epoch % conf["INTERVAL_SAVE"] == 0 or conf["KEEP_SAVE"] and epoch >= conf["KEEP_SAVE"]:
                save_path=os.path.join(weights_path, 'epoch_{}.pth'.format(epoch))
                torch.save(ckpt, save_path)
                logger.info('saving epoch {} weights file to {}'.format(epoch,save_path))

    if writer:
        writer.close()

    # 如果是使用多进程训练, 那么销毁进程组
    if WORLD_SIZE > 1 and RANK == 0:
        logger.info('Destroying process group... ')
        dist.destroy_process_group()


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/cifar100.yaml", help='experiment config file')
    parser.add_argument('--pre-weight', type=str, default="", help='Pre-training weight')
    parser.add_argument('--freeze', type=str, nargs="+", default=[], help='freeze layer name, use space separate layer name.')
    parser.add_argument('--resume', type=str, default="", help='The model path of recovery training')
    parser.add_argument('--sync_bn', action="store_true", help='Use SyncBatchNorm for DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, don’t to modify')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parser_opt()
    main(args)

