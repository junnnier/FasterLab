import torch
import torchvision
# --------本地导入--------
from extension.detr_module.tools import PostProcess
from extension.detr_module.segmentation import PostProcessSegm
from extension.detr_module.coco_eval import CocoEvaluator
from extension.detr_module.misc import MetricLogger, SmoothedValue, reduce_dict


class EvaluateFunction(object):
    def __init__(self, param):
        self.param = param
        self.postprocessors = {'bbox': PostProcess()}
        if param["masks"]:
            self.postprocessors['segm'] = PostProcessSegm()
        self.metric_logger = MetricLogger(delimiter="  ")
        self.metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))

    def __call__(self, model, data_loader, loss_func, device, config, save_dir="./"):
        if loss_func:
            loss_func.eval()

        base_ds = self.get_coco_api_from_dataset(data_loader.dataset)
        iou_types = tuple(k for k in ('segm', 'bbox') if k in self.postprocessors.keys())
        coco_evaluator = CocoEvaluator(base_ds, iou_types)

        # 对测试集进行验证
        for images, labels in self.metric_logger.log_every(data_loader, 10, 'Test:'):
            images = images.to(device)
            labels = [{key: value.to(device) for key, value in label.items()} for label in labels]

            outputs = model(images)

            if loss_func:
                loss_dict, loss_item = loss_func(outputs, labels)
                weight_dict = loss_func.weight_dict
                loss_dict_reduced = reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
                loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
                self.metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
                self.metric_logger.update(class_error=loss_dict_reduced['class_error'])

            orig_target_sizes = torch.stack([t["orig_size"] for t in labels], dim=0)
            results = self.postprocessors['bbox'](outputs, orig_target_sizes)
            if 'segm' in self.postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in labels], dim=0)
                results = self.postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(labels, results)}

            if coco_evaluator is not None:
                coco_evaluator.update(res)
        self.metric_logger.synchronize_between_processes()
        print("Averaged stats:", self.metric_logger)

        if coco_evaluator is not None:
            coco_evaluator.synchronize_between_processes()

        # 累计所有预测图像的结果
        if coco_evaluator is not None:
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

        stats = {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}
        if coco_evaluator is not None:
            if 'bbox' in self.postprocessors.keys():
                stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            if 'segm' in self.postprocessors.keys():
                stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

        return stats["coco_eval_bbox"][0], stats["class_error"]

    def get_coco_api_from_dataset(self, dataset):
        for _ in range(10):
            if isinstance(dataset, torch.utils.data.Subset):
                dataset = dataset.dataset
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            return dataset.coco