import logging
import os
import sys


def setup_logger(log_dir=None, filename="logs.txt", rank=-1):
    child_logger = logging.getLogger(name=__name__)  # 如果传递了name参数，得到的就是一个非root的Logger
    child_logger.propagate = False  # 不将此子日志记录器的消息传播给根日志记录器
    # 如果当前rank不是主进程，不对其构建handler
    if rank > 0:
        return child_logger
    # 设置日志记录等级, DEBUG < INFO < WARNING < ERROR < CRITICAL
    child_logger.setLevel(logging.INFO)
    # 日志输出到文件
    if log_dir:
        log_path = os.path.join(log_dir, filename)
        file_handler = logging.FileHandler(filename=log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S'))
        child_logger.addHandler(file_handler)
    # 日志输出到屏幕
    stream_handler = logging.StreamHandler(sys.stdout)
    child_logger.addHandler(stream_handler)
    return child_logger