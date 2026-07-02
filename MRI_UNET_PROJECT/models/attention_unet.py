# ============================================================
# U-Net / Attention U-Net模型
# ============================================================

"""
定义模型结构，不包含训练、验证等过程

思路（或者说参考）来源：
1.经典U-Net模型的编码器-解码器结构：下采样卷积池化提取语义特征，上采样反卷积恢复空间分辨率；
2.（可选）经典Attention U-Net的注意力门控思想：在 skip connection 前过滤 encoder特征，尽量抑制无关背景区域，突出可能的病灶区域。

本项目情况：
1. 输入为单通道 MRI，默认in_ch=1；
2. 输出为单通道 logits，后续再经过 sigmoid（化为 0~1）和 threshold（>则认为 1，<则认为 0) 转为二值 mask；
"""

import torch
import torch.nn as nn


def make_group_norm(num_channels,max_groups=8):
    """
    小 batch 医学分割中，用 GroupNorm 替代 BatchNorm 以提升训练稳定性。
    根据通道数自动选择 GroupNorm 的组数。

    参数：
        num_channels: 当前特征图通道数
        max_groups: 最多分多少组，默认不超过 8 组
    """
    for g in range(min(max_groups,num_channels),0,-1):
        if num_channels % g == 0:
            return nn.GroupNorm(g,num_channels)

    # 理论上不会走到这里；保留兜底逻辑，等价于LayerNorm
    return nn.GroupNorm(1,num_channels)


class DoubleConv(nn.Module):
    """
    双层卷积块。
    Conv 3×3 -> GroupNorm -> ReLU
    Conv 3×3 -> GroupNorm -> ReLU
    """

    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1,bias=False),
            make_group_norm(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1,bias=False),
            make_group_norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    注意力门控模块。

    输入：
        x: encoder 侧传来的 skip feature，空间细节多，包含背景噪声；
        g: decoder 侧传来的 gating feature，语义信息更强，用来指导关注位置。


    先用 1×1 卷积把 x 和 g 压到相同的中间通道数；
    再相加、ReLU、Sigmoid 得到 attention map；
    加权输出 x * attention_map。
    """

    def __init__(self,x_ch,g_ch,mid_ch):
        super().__init__()

        # encoder 分支：把 skip feature 映射到中间通道
        self.Wx = nn.Sequential(
            nn.Conv2d(x_ch,mid_ch,kernel_size=1,stride=1,padding=0,bias=False),
            make_group_norm(mid_ch),
        )

        # decoder 分支：把 gating feature 映射到中间通道
        self.Wg = nn.Sequential(
            nn.Conv2d(g_ch,mid_ch,kernel_size=1,stride=1,padding=0,bias=False),
            make_group_norm(mid_ch),
        )

        # 生成单通道注意力图，0~1
        self.psi = nn.Sequential(
            nn.Conv2d(mid_ch,1,kernel_size=1,stride=1,padding=0,bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x,g):
        x_conv = self.Wx(x)
        g_conv = self.Wg(g)

        #融合，生成注意力权重
        combined = self.relu(x_conv + g_conv)
        atten_map = self.psi(combined)
        return x * atten_map

class SkipIdentity(nn.Module):
    """
    关闭 Attention 时使用的占位模块。
    基础 U-Net 不需要根据 g 生成注意力图，因此直接返回 x，
    相当于普通 U-Net 中的原始 skip connection。
    """
    def forward(self, x, g):
        # 不做注意力筛选，直接把 encoder 的 skip feature 传给 decoder。
        return x


class AttentionUNet(nn.Module):
    """
    Attention U-Net。

    Encoder:
    4个下采样块  enc1 -> enc2 -> enc3 -> enc4 -> bottleneck

    Decoder:
    4个上采样块
    up4 -> attention(e4, d4) -> dec4
    up3 -> attention(e3, d3) -> dec3
    up2 -> attention(e2, d2) -> dec2
    up1 -> attention(e1, d1) -> dec1

    Output:  1x1 conv -> 1通道 logits
    """

    def __init__(self,in_ch=1,out_ch=1,base=32,use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # Encoder逐层下采样
        self.enc1 = DoubleConv(in_ch,base)
        self.enc2 = DoubleConv(base,base * 2)
        self.enc3 = DoubleConv(base * 2,base * 4)
        self.enc4 = DoubleConv(base * 4,base * 8)
        self.bottleneck = DoubleConv(base * 8,base * 16)

        #最大池化降低空间分辨率
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base * 16,base * 8,kernel_size=2,stride=2)
        self.att4 = AttentionGate(base * 8,base * 8,base * 4) if use_attention else SkipIdentity()
        self.dec4 = DoubleConv(base * 16,base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8,base * 4,kernel_size=2,stride=2)
        self.att3 = AttentionGate(base * 4,base * 4,base * 2) if use_attention else SkipIdentity()
        self.dec3 = DoubleConv(base * 8,base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4,base * 2,kernel_size=2,stride=2)
        self.att2 = AttentionGate(base * 2,base * 2,base)  if use_attention else SkipIdentity()
        self.dec2 = DoubleConv(base * 4,base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2,base,kernel_size=2,stride=2)
        self.att1 = AttentionGate(base, base,max(base // 2, 1))  if use_attention else SkipIdentity()
        self.dec1 = DoubleConv(base * 2,base)

        # 1×1卷积把特征通道压成最终 mask 通道数
        self.head = nn.Conv2d(base,out_ch,kernel_size=1)

    def forward(self, x):
        # Encoder

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # Decoder + Attention skip connection

        d4 = self.up4(b)
        a4 = self.att4(e4,d4)
        d4 = self.dec4(torch.cat([d4,a4],dim=1))

        d3 = self.up3(d4)
        a3 = self.att3(e3,d3)
        d3 = self.dec3(torch.cat([d3,a3],dim=1))

        d2 = self.up2(d3)
        a2 = self.att2(e2,d2)
        d2 = self.dec2(torch.cat([d2,a2],dim=1))

        d1 = self.up1(d2)
        a1 = self.att1(e1,d1)
        d1 = self.dec1(torch.cat([d1,a1],dim=1))

        # 输出 logits
        return self.head(d1)


def count_params(m):
    """计算模型可训练参数数量，用于在训练日志中记录训练规模 """
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
