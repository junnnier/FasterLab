# FasterLab
A deep learning model training platform with high scalability, everything is modular.
## Support
Currently supported model structures are:
```
Classification:
    ResNet、VGG、MobileNet、GoogleNet、InceptionNet、RepVGG
Object Detection:
    YOLOv5、DETR
Segmentation:
    DeeplabV3+
Image-Text:
    CLIP
DeNoising:
    EfficientDeRain
OCR:
    DBNet、RCNN
```
## Instructions
You can customize various modules for your training tasks, such as: **network structures** (under ./models folder), **network weight initialize** (under ./utils/weight_initialize.py file), **dataset loading** (under ./dataset folder), **optimizers** (under ./utils/optimizer.py file), **schedulers** (under ./utils/scheduler.py file), **training process** (under ./training folder), **loss function** (under ./loss folder), **evaluate function** (under ./evaluate folder), **inference pre-process** (under ./inference_pre folder), **inference post-process** (under ./inference_post folder), and more.
### 1. config file
File format reference **xxx.yaml** example under **./config** folder. The parameters in the configuration file can be increased or decreased, you can fill in the required parameters according to different training tasks or custom function modules, everything is modular. The parameter format must be specified as required, for example:

When you set to "Adam" under **OPTIMIZER**, build a dictionary with the same name as "Adam" under **OPTIMIZER_PARAM**, where you can define the parameters you need to pass in, such as "lr", "momentun", and so on. Other **XXXX** modules do the same, building parameters under the corresponding name **XXXX_PARAM**.
```
OPTIMIZER: Adam
OPTIMIZER_PARAM:
    Adam:
        lr: 0.001
        momentum: 0.937
```
### 2. network structures
You need to create a file about the structure of the model, where the file name must be the same as the parameter of the variable **MODEL_NAME** in the configuration file, and the file must contain the function name in the following example as the starting call function.
```
def create_model(param):
    ...
    return ResNet()
```
### 3. network weight initialize
You need to create a function in the file, where the function name **xxx** must be the same as the parameter of the variable **INITIALIZE_WEIGHTS** in the configuration file, and the call function template in the file is as follows.
```
def xxx(model, param):
    ...
```
### 4. dataset loading
In order to adapt to different training tasks, you can customize the dataset loading module. You need to create a function in the file, where the file name must be the same as the parameter of the variable **DATASET_TASK** in the configuration file, and the call function template in the file is as follows.

If you want to use the default dataset loading module, you need to generate the specified file format. Please refer to the [Dataset Format](##Dataset Format) for details.
```
from dataset.Base import BaseDataset

class DatasetTask(BaseDataset):
    def __init__(self, conf, describe=""):
        ...

    def __getitem__(self, index):
        ...        
        return

    def __len__(self):
        return
```
### 5. optimizers
You need to create a function in the file, where the function name **xxx** must be the same as the parameter of the variable **OPTIMIZER** in the configuration file, and the call function template in the file is as follows.
```
def xxx(net, param):
    ...
    return optimizer
```
### 6. schedulers
You need to create a function in the file, where function name **xxx** must be the same as the parameter of the variable **SCHEDULER** in the configuration file, and the call function template in the file is as follows.
```
def xxx(optimizer, param):
    ...
    return train_scheduler
```
### 7. training process
You need to create a file that defines the training process, where the file name must be the same as the parameter of the variable **TRAINING_FUNCTION** in the configuration file, and the call function template in the file is as follows.
```
from training.Base import BaseTrain

class EpochTrain(BaseTrain):
    def __init__(self, param):
        super(EpochTrain, self).__init__()

    def __call__(self, net, train_loader, optimizer, device, epoch, total_epoch, RANK, loss_function, logger, writer):
        ...
```
### 8. evaluate function
You need to create a file that defines the evaluate process, where the file name must be the same as the parameter of the variable **EVALUATE_FUNCTION** in the configuration file, and the call function template in the file is as follows.
```
class EvaluateFunction(object):
    def __init__(self, param):
        ...

    def __call__(self, model, data_loader, loss_func, device, config, save_dir="./"):
        ...
        return 验证精度, 损失值列表
```
### 9. inference pre-process
If you need to test a trained model, you can to create a file that defines pre-processing during model inference, and the call function template in the file is as follows.
```
from utils.image_tools import load_image

class PreTreatment(object):
    def __init__(self):
        ...

    def __call__(self, image_path, image_size, force_resize, device):
        # 读取图片
        image = load_image(image_path)
        ...
        return image, image_shapes
```
### 10. inference post-process 
If you need to test a trained model, you can to create a file that defines post-processing during model inference, and the call function template in the file is as follows.
```
from utils.torch_tools import time_sync

class PostProcess(object):
    def __init__(self):
        ...

    def __call__(self, output, shapes, config, image_path, save):
        ...
        st = time_sync()
        ...  # 模型推理
        et = time_sync() - st
        # 打印信息
        result_info = ""
        print(result_info)
        return result_info, et
```
## Command
### 1. train
```bash
python train.py --config config/repvgg.yaml
```
```bash
python train.py --resume runs/cifar100_xxxxxxxx_xxxxxx/weights/last.pth
```
```bash
python -m torch.distributed.launch --master_port 9999 --nproc_per_node=2 train.py --config config/repvgg.yaml
```
### 2. validation
```bash
python val.py --weight runs/cifar100_xxxxxxxx_xxxxxx/weights/best.pth --config runs/cifar100_xxxxxxxx_xxxxxx/config.yaml
```
### 3. test
Parameter --postprocess is set according to the prediction task.
```bash
python test.py --weight runs/cifar100_xxxxxxxx_xxxxxx/weights/best.pth --config config/repvgg.yaml --image-path test_image_list.txt --postprocess ClassifyPostProcess --save test_result.txt
```
## Dataset Format
### ClassifyDataset
The data includes train.txt and test.txt. The contents of the file contain the absolute path and label name of the image in the following format. The absolute paths and labels separated by "\t".
```
/dataset/train/xxx.png      label_1
/dataset/train/xxx.png      label_2
...
/dataset/train/xxx.png      label_3
```
### DeRainDataset
The data includes train.txt and test.txt. The contents of this file contain the absolute path of the input image and the label image in the following format, separated by "\t".
```
/dataset/train/rain/xxx.png      /dataset/train/norain/xxx.png
/dataset/train/rain/xxx.png      /dataset/train/norain/xxx.png
...
/dataset/train/rain/xxx.png      /dataset/train/norain/xxx.png
```
### CLIPDataset
The data includes train.txt and test.txt. The contents of this file contain the absolute path of the input image and the image prompt words and the label name of the image in the following format, separated by "\t".
```
/dataset/train/xxx.png      A photo of an apple on the desk .      apple
/dataset/train/xxx.png      A photo of an orange on the desk .      orange
...
/dataset/train/xxx.png      A photo of an pear on the desk .      pear
```
### DetectDataset
The data includes train.txt and test.txt. The contents of this file contain the absolute path of the input image in the following format.
```
/dataset/train/xxx.png
/dataset/train/xxx.png
...
/dataset/train/xxx.png
```
The label file format of each image is as follows: Each line is a target box in the image. The data represents class_num, x_center, y_center, width and height respectively, separated by space. The value of class_num starts from 0, and the four values of the box are the number between 0 and 1 normalized with respect to the image resolution size. That is, the upper left corner is (0,0), the lower right corner is (1,1), and if there is no target box in the picture, it is an empty file.
```
20 0.5 0.4 0.8 0.3
15 0.4 0.2 0.5 0.6
```
### OCRRecognizeDataset
The data includes train.txt and test.txt. The contents of the file contain the absolute path and character label index of the image in the following format. The absolute paths and labels separated by "\t", the character indexes are separated by Spaces.
```
/dataset/train/xxx.png      89 201 241 178 19 94 19 22 26 656
/dataset/train/xxx.png      120 1061 2 376 78 249 272 272 120 1061
...
/dataset/train/xxx.png      923 1229 1328 337 21 2 1130 153 522 9
```