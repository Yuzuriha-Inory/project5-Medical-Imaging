# ============================================================
# 数据集读取与预处理 + 训练阶段数据增强
# 注：
#   1.这一板块负责LGG-MRI单张切片的读取、同步增强、归一化和tensor转换。
#   2.训练集启用温和增强；同时验证集和测试集只做resize 与归一化，避免评估结果受随机增强影响。
#   3.所有几何变换会同时作用于image和mask。
#   4.MRI图像按单通道灰度图读取，对非零脑区进行z-score归一化，来减少黑色背景对强度分布的影响。

# ============================================================

import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

from config import (
    AUG_HFLIP_PROB,
    AUG_ROTATE_DEGREES,
    AUG_TRANSLATE_FRAC,
    AUG_SCALE_RANGE,
    AUG_INTENSITY_PROB,
    AUG_BRIGHTNESS_RANGE,
    AUG_CONTRAST_RANGE,
    AUG_GAMMA_PROB,
    AUG_GAMMA_RANGE,
    AUG_BLUR_PROB,
    AUG_BLUR_RADIUS_RANGE,
    AUG_NOISE_PROB,
    AUG_NOISE_STD_RANGE,
)

class LGGDataset(Dataset):
    """
        LGG 脑部 MRI 二值分割数据集。

        这类只负责单张切片级的数据读取和预处理：
            1.读取 MRI 图像和对应 mask；
            2.将图像统一 resize 到指定尺寸；
            3.训练阶段做温和数据增强；
            4.对 MRI 非零脑区做 z-score 归一化；
            5.返回 PyTorch 训练需要的 tensor。

        返回：
            image:FloatTensor,shape = (1, H, W)
            mask :FloatTensor,shape = (1, H, W),取值为 0 或 1
    """

    def __init__(self,samples,augment=False,img_size=256):
        """
            参数：
                samples:
                    [(img_path, mask_path), ...] 形式的列表。
                    样本的 train/val/test 划分已经在 data_split.py 中按病例完成。

                augment:
                    是否启用训练增强。训练集为 True，验证集和测试集应为 False。

                img_size:
                    统一后的图像尺寸，默认使用 256。
        """
        self.samples=samples
        self.augment=augment
        self.img_size=img_size

    def __len__(self):
        return len(self.samples)

    def _resize_pair(self,img,mask):
        """
        同步 resize MRI 和 mask。

        MRI 图像使用双线性插值，保证灰度变化相对平滑；
        mask 使用最近邻插值，保证标签仍然是离散的 0/1。
        """
        target_size = [self.img_size, self.img_size]

        img = TF.resize(
            img,
            target_size,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            target_size,
            interpolation=TF.InterpolationMode.NEAREST,
        )
        return img, mask

    def _random_flip(self,img,mask):
        """
        随机水平翻转。不使用 vertical flip。
        因为脑部 MRI 有固定解剖方向，上下翻转可能产生不符合真实分布的样本。
        """
        if random.random() < AUG_HFLIP_PROB:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        return img, mask

    def _random_affine(self,img,mask):
        """
        轻微仿射增强：小角度旋转、少量平移和轻微缩放。
        用于模拟扫描位置和裁剪上的轻微差异。但参数不宜过大，否则可能破坏医学图像的解剖结构。
        """
        width,height = img.size

        max_dx = int(width * AUG_TRANSLATE_FRAC)
        max_dy = int(height * AUG_TRANSLATE_FRAC)

        angle = random.uniform(-AUG_ROTATE_DEGREES,AUG_ROTATE_DEGREES)

        translate = (
            random.randint(-max_dx,max_dx),
            random.randint(-max_dy,max_dy),
        )
        scale = random.uniform(AUG_SCALE_RANGE[0],AUG_SCALE_RANGE[1])
        shear = 0.0

        img = TF.affine(
            img,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=0,
        )
        mask = TF.affine(
            mask,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=TF.InterpolationMode.NEAREST,
            fill=0,
        )
        return img, mask

    def _intensity_augment(self, img):
        """
        MRI 强度增强，只作用于图像，不作用于 mask。
        亮度、对比度和 gamma 变化用于模拟不同病例、不同扫描条件下的灰度差异。
        这些增强保持较弱，避免破坏小病灶边界，可能能够增强模型鲁棒性。
        """

        if random.random() < AUG_INTENSITY_PROB:
            # 设置亮度增强
            brightness = random.uniform(
                AUG_BRIGHTNESS_RANGE[0],
                AUG_BRIGHTNESS_RANGE[1],
            )
            #设置对比度增强
            contrast = random.uniform(
                AUG_CONTRAST_RANGE[0],
                AUG_CONTRAST_RANGE[1],
            )
            img = TF.adjust_brightness(img,brightness_factor=brightness)
            img = TF.adjust_contrast(img,contrast_factor=contrast)

        #设置gamma变化
        if random.random() < AUG_GAMMA_PROB:
            gamma = random.uniform(AUG_GAMMA_RANGE[0],AUG_GAMMA_RANGE[1])
            img = TF.adjust_gamma(img,gamma=gamma,gain=1.0)

        return img


    def _blur_if_needed(self,img):
        """
        低概率加入轻微模糊。
        模糊半径设置得很小，只是模拟成像质量上的轻微变化。如果半径过大，会让病灶边界变得不真实。
        """
        if random.random() < AUG_BLUR_PROB:
            radius = random.uniform(
                AUG_BLUR_RADIUS_RANGE[0],
                AUG_BLUR_RADIUS_RANGE[1],
            )
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        return img

    def _joint_transform(self,img,mask):
        """
            对 image 和 mask 做同步几何处理。
            训练集：
                resize + 水平翻转 + 轻微仿射变换 + 图像强度增强 + 轻微模糊
            验证集 / 测试集：
                只做 resize，不做随机增强。
        """
        img,mask = self._resize_pair(img,mask)

        if not self.augment:
            return img,mask

        img,mask = self._random_flip(img,mask)
        img,mask = self._random_affine(img,mask)

        # 强度增强和模糊只改MRI图像，不改mask。
        img = self._intensity_augment(img)
        img = self._blur_if_needed(img)

        return img,mask

    @staticmethod
    def _normalize_mri(img_arr):
        """
        对单张 MRI 做非零脑区 z-score 归一化。

        数据中黑色背景占比较大，如果直接对整张图求均值和方差，
        背景会影响强度分布。因此这里只使用非零脑区估计均值和标准差。
        """
        brain = img_arr[img_arr > 0.02]

        if brain.size > 0:
            mean = float(brain.mean())
            std = float(brain.std()) + 1e-6
            img_arr = (img_arr - mean) / std
        else:
            # 极少数异常图像如果没有明显非零区域，则使用一个保守归一化方式。
            img_arr = (img_arr - 0.5) / 0.5

        return img_arr

    def _add_noise_if_needed(self,img_arr):
        """
        在 z-score 后加入轻微高斯噪声。
        只在训练阶段使用。噪声强度很小，用于提高模型对灰度扰动的鲁棒性。
        """
        if not self.augment:
            return img_arr

        if random.random() >= AUG_NOISE_PROB:
            return img_arr

        noise_std = random.uniform(
            AUG_NOISE_STD_RANGE[0],
            AUG_NOISE_STD_RANGE[1],
        )
        noise = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=img_arr.shape,
        ).astype(np.float32)

        return img_arr + noise

    def __getitem__(self,idx):
        """
        读取一个样本，并转换为模型输入格式。
        """
        img_path,mask_path = self.samples[idx]

        # MRI-image按灰度图读取；mask也按灰度图读取，之后再二值化。
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img,mask = self._joint_transform(img,mask)

        # 图像先转成0~1的float数组，再做MRI强度归一化、添加噪声干扰
        img_arr = np.array(img).astype(np.float32) / 255.0
        img_arr = self._normalize_mri(img_arr)
        img_arr = self._add_noise_if_needed(img_arr)

        # 限制极端值，防止少数异常灰度影响训练稳定性。
        img_arr = np.clip(img_arr,-5.0,5.0)

        # mask只保留0/1，作为二值分割标签。
        mask_arr = (np.array(mask) > 0).astype(np.float32)

        img_t = torch.from_numpy(img_arr).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0).float()

        return img_t, mask_t
