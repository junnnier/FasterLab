import os
import cv2
import numpy as np
import math
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
# --------本地导入--------
from utils.visual import derain_image_save


class EvaluateFunction(object):
    def __init__(self, param):
        self.visual = param["visual"]

    def __call__(self, model, data_loader, loss_func, device, config, save_dir=""):
        test_avg_loss = 0.0
        psnr_sum, ssim_sum = 0, 0
        describe = ('%10s' + '%10s' * 2) % ("class", "psnr_avg", "ssim_avg")
        data_bar = tqdm(enumerate(data_loader), desc=describe, total=len(data_loader))

        # 对验证集进行预测
        for batch_index, (true_input, true_target, shapes) in data_bar:
            true_input = true_input.to(device)
            true_target = true_target.to(device)
            # 推理
            fake_target = model(true_input)
            # 计算loss
            if loss_func:
                total_loss, loss_item = loss_func(fake_target, true_target)
                test_avg_loss = (test_avg_loss * batch_index + total_loss.item()) / (batch_index + 1)
            # 是否可视化图片
            if self.visual and save_dir:
                img_list = [true_input, fake_target, true_target]
                name_list = ['input', 'predict', 'gt']
                derain_image_save(sample_folder=save_dir, sample_name="{}".format(batch_index+1), img_list=img_list, name_list=name_list, pixel_max_cnt=255, height=shapes[0][0], width=shapes[0][1])
            # 计算差异度
            img_pred_recover = self.recover_process(fake_target, height=shapes[0][0], width=shapes[0][1])
            img_gt_recover = self.recover_process(true_target, height=shapes[0][0], width=shapes[0][1])
            psnr_sum = psnr_sum + self.psnr(img_pred_recover, img_gt_recover)
            ssim_sum = ssim_sum + compare_ssim(img_gt_recover, img_pred_recover, channel_axis=2, multichannel=True, data_range=255)

        print(("%10s" + "%10.3g" * 2) % ("all", psnr_sum/len(data_loader), ssim_sum/len(data_loader)))
        fitness_acc = (psnr_sum + ssim_sum) / len(data_loader)

        if save_dir:
            result_save_path = os.path.join(save_dir, "val_result.txt")
            with open(result_save_path, "w", encoding="utf-8") as f:
                f.write("psnr average: {}\tssim average: {}\n".format(psnr_sum/len(data_loader), ssim_sum/len(data_loader)))
            print("The validation result save to:", result_save_path)

        return fitness_acc, test_avg_loss

    def recover_process(self, img, height=-1, width=-1):
        height = height if isinstance(height,int) else height.item()
        width = width if isinstance(width,int) else width.item()
        img = img * 255.0
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, 255)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = img_copy.astype(np.float32)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        if (height != -1) and (width != -1):
            img_copy = cv2.resize(img_copy, (width, height))
        return img_copy

    def psnr(self, pred, target):
        mse = np.mean((pred - target) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))