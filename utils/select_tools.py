import os
import sys
import importlib
import torch


def get_dataset_task(conf):
    """
    Args:
        conf: 配置文件
    Returns: 构建的数据集加载函数
    """
    task_name = conf["DATASET_TASK"]
    try:
        dataset_package = importlib.import_module("dataset.{}".format(task_name))
        dataset_task = getattr(dataset_package, "DatasetTask")
    except AttributeError:
        print('error: the dataset deal with function name you have entered is not supported yet, '
              'please create "{}.py" file in directory of Dataset/.'.format(task_name))
        sys.exit()
    return dataset_task


def get_network(conf):
    """
    Args:
        conf: 配置文件
    Returns: 构建好的网络模型
    """
    model_name = conf["MODEL_NAME"]
    param = conf["MODEL_NAME_PARAM"].get(model_name, None)
    try:
        model = importlib.import_module("models.{}".format(model_name))
    except ModuleNotFoundError:
        print('error: the model name you have entered is not supported yet, please create "{}" in directory of models/.'.format(model_name))
        sys.exit()
    else:
        net = model.create_model(param)
    return net


def get_loss_function(net, conf):
    """
    Args:
        net: 训练的网络
        conf: 配置文件
    Returns: 构建的损失函数
    """
    loss_name = conf["LOSS_FUNCTION"]
    param = conf["LOSS_FUNCTION_PARAM"].get(loss_name, None)
    try:
        loss_package = importlib.import_module("loss.{}".format(loss_name))
        loss_fun = getattr(loss_package, "LossFunction")
    except AttributeError:
        print('error: the loss function name you have entered is not supported yet, '
              'please create "{}.py" file in directory of loss/.'.format(loss_name))
        sys.exit()
    else:
        loss_fun = loss_fun(net, param)
    return loss_fun


def get_optimizer(net,conf):
    """
    Args:
        net: 训练的网络
        conf: 配置文件
    Returns: 构建的优化器
    """
    opti_name = conf["OPTIMIZER"]
    param = conf["OPTIMIZER_PARAM"].get(opti_name, None)
    try:
        optimizer_package = importlib.import_module("utils.optimizer")
        optimizer = getattr(optimizer_package, opti_name)
    except AttributeError:
        print('error: the optimizer name you have entered is not supported yet, '
              'please create "{}" in file of utils/optimizer.py.'.format(opti_name))
        sys.exit()
    else:
        optimizer = optimizer(net, param)
    return optimizer


def get_scheduler(optimizer,conf):
    """
    Args:
        optimizer: 优化器
        conf: 配置文件
    Returns: 构建好的优化器
    """
    scheduler_name = conf["SCHEDULER"]
    param = conf["SCHEDULER_PARAM"].get(scheduler_name, None)
    try:
        scheduler_package = importlib.import_module("utils.scheduler")
        train_scheduler = getattr(scheduler_package, scheduler_name)
    except AttributeError:
        print('error: the scheduler name you have entered is not supported yet, '
              'please create "{}" in file of utils/scheduler.py.'.format(scheduler_name))
        sys.exit()
    else:
        train_scheduler = train_scheduler(optimizer, param)
    return train_scheduler


def get_initialize_weights(net, conf):
    """
    Args:
        net: 网络模型
        conf: 配置文件
    Returns: 初始化权重后的网络
    """
    mode = conf["INITIALIZE_WEIGHTS"]
    param = conf["INITIALIZE_WEIGHTS_PARAM"].get(mode, None)
    try:
        torch_package = importlib.import_module("utils.weight_initialize")
        initialize = getattr(torch_package, mode)
    except AttributeError:
        print('error: the model initialize name you have entered is not supported yet, '
              'please create "{}" in file of utils/weight_initialize.py.'.format(mode))
        sys.exit()
    else:
        initialize(net, param)
        print("Use {} initialize weights.".format(mode))


def get_evaluate_function(conf):
    """
    Args:
        conf: 配置文件
    Returns: 构建的解析函数
    """
    evaluate_name = conf["EVALUATE_FUNCTION"]
    param = conf["EVALUATE_FUNCTION_PARAM"].get(evaluate_name, None)
    try:
        evaluate_package = importlib.import_module("evaluate.{}".format(evaluate_name))
        evaluate_fun = getattr(evaluate_package, "EvaluateFunction")
    except AttributeError:
        print('error: the loss function name you have entered is not supported yet, '
              'please create "{}.py" file in directory of evaluate/.'.format(evaluate_name))
        sys.exit()
    else:
        evaluate_fun = evaluate_fun(param)
    return evaluate_fun


def get_training_function(conf):
    """
    Args:
        conf: 配置文件
    Returns: 构建的解析函数
    """
    training_name = conf["TRAINING_FUNCTION"]
    param = conf["TRAINING_FUNCTION_PARAM"].get(training_name, None)
    try:
        training_package = importlib.import_module("training.{}".format(training_name))
        training_fun = getattr(training_package, "EpochTrain")
    except AttributeError:
        print('error: the training function name you have entered is not supported yet, '
              'please create "{}.py" file in directory of training/.'.format(training_name))
        sys.exit()
    else:
        training_fun = training_fun(param)
    return training_fun


def get_pretreatment_function(pretreatment_name):
    """
    Args:
        pretreatment_name: 预处理函数名
    Returns: 预处理函数对象
    """
    try:
        pretreatment_package = importlib.import_module("inference_pre.{}".format(pretreatment_name))
        pretreatment_operate = getattr(pretreatment_package, "PreTreatment")
    except AttributeError:
        print('error: the pretreatment name you have entered is not supported yet, '
              'please create "{}.py" file in directory of inference_pre/.'.format(pretreatment_name))
        sys.exit()
    else:
        pretreatment_operate = pretreatment_operate()
    return pretreatment_operate


def get_postprocess_function(postprocess_name):
    """
    Args:
        postprocess_name: 后处理函数名
    Returns: 后处理函数对象
    """
    try:
        postprocess_package = importlib.import_module("inference_post.{}".format(postprocess_name))
        postprocess_operate = getattr(postprocess_package, "PostProcess")
    except AttributeError:
        print('error: the postprocess name you have entered is not supported yet, '
              'please create "{}.py" file in directory of inference_post/.'.format(postprocess_name))
        sys.exit()
    else:
        postprocess_operate = postprocess_operate()
    return postprocess_operate


def select_device(device, batch_size=None):
    s = f'torch {torch.__version__} '  # 打印torch版本
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制torch.cuda.is_available() = False
        s += " cpu"
    else:
        # 检查batch_size是否能被device_count整除
        device_list = device.split(',')
        n = len(device_list)
        if n > 1 and batch_size:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        # 设置环境变量，可见的cuda设备
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        # 检查cuda是否可用
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'
        # 获取每个gpu的属性
        for i, d in enumerate(device_list):
            p = torch.cuda.get_device_properties(i)
            s += f" CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)"  # 将bytes转换为MB
    print(s)
    return torch.device("cpu" if cpu else "cuda:0")
