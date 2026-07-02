# ============================================================
# 损失函数
# ------------------------------------------------------------
# 注：
#   1.这一板块负责构造分割任务中使用的3种不同loss
#   2.3组实验：BCE、Dice、BCE + Dice
#   3.模型输出为 logits，因此 BCE 部分使用 BCEWithLogitsLoss，Dice部分会在内部先做 sigmoid。
# ============================================================

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss，用于直接优化预测区域和真实区域的重叠程度。对医学小目标分割来说，Dice 比单纯 BCE 更关注前景区域。
    但 Dice Loss 过于积极的话，有时会在无病灶切片上产生一些误报，
    因此这里额外加入很轻的背景正则项，用来压制背景区域的预测概率。
    """

    def __init__(self,smooth=1e-6,bg_reg=0.02):
        super().__init__()

        self.smooth = smooth #smooth避免分母为0，同时让极小区域计算更稳定

        self.bg_reg = bg_reg ## 背景正则权重不宜过大

    def forward(self,logits,target):
        """
        logits: 模型原始输出，形状通常为 (B, 1, H, W)
        target: 二值 mask，取值为 0/1，形状与 logits 一致
        """

        prob = torch.sigmoid(logits) #0~1概率

        # 按batch拉平，逐样本计算Dice
        prob_flat = prob.flatten(1)
        target_flat = target.flatten(1)

        inter = (prob_flat * target_flat).sum(1) #重叠部分
        denom = prob_flat.sum(1) + target_flat.sum(1)  #Dice分母

        #Dice与Dice Loss计算式
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1 - dice.mean()

        # 背景正则，惩罚真实背景区域上的预测概率，减少无病灶切片上的零散假阳性。
        bg_loss = (prob_flat * (1 - target_flat)).mean()

        return dice_loss + self.bg_reg * bg_loss


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice 联合损失。
    BCE 稳定像素级二分类学习；
    Dice 强化预测区域与真实病灶区域的重叠。
    两者组合通常比单独使用某一个 loss 更稳。
    """

    def __init__(self,smooth=1e-6,w_bce=0.45,w_dice=0.55,pos_weight=4.0):
        super().__init__()

        self.smooth = smooth
        self.w_bce = w_bce
        self.w_dice = w_dice

        self.register_buffer("pos_weight",torch.tensor([float(pos_weight)]))

        self.dice = DiceLoss(smooth=smooth)

    def forward(self,logits,target):
        # pos_weight用于放大病灶像素的损失权重，缓解前景像素过少的问题。
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight.to(logits.device),
        )

        dice_loss = self.dice(logits,target)

        # 当前设置略微偏向Dice，表明此时更关心病灶区域重叠质量。
        return self.w_bce * bce_loss + self.w_dice * dice_loss


def get_loss(name,pos_weight,device):
    """
    根据实验名称返回对应损失函数。

    name: 实验名称，对应 config.py 中的 EXPERIMENTS
    pos_weight: 从训练集 mask 自动估计得到的正类像素权重
    device: 当前训练设备，用于将 BCE 的 pos_weight 放到正确设备上

    返回：
        可直接用于训练的 PyTorch loss 。
    """
    if name == "unet_bce":
        # 单独BCE实验：用于观察像素级二分类损失的表现
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([float(pos_weight)],device=device)
        )

    if name == "unet_dice":
        # 单独Dice实验：用于观察区域重叠优化的表现
        return DiceLoss(smooth=1e-6,bg_reg=0.02)

    if name == "unet_bce_dice":
        # 联合损失实验：BCE + Dice
        return BCEDiceLoss(
            smooth=1e-6,
            w_bce=0.45,
            w_dice=0.55,
            pos_weight=pos_weight,
        )

    raise ValueError(f"未知的 loss 名称: {name}")


