# --------本地导入--------
from training.BaseAMP import BaseAMPTrain


class EpochTrain(BaseAMPTrain):
    def __init__(self, param):
        super(EpochTrain, self).__init__(param)
        # 设置打印抬头
        self.title_string = ("%10s" * 6) % ("Epoch", "mem", "loss-avg", "box", "obj", "class")