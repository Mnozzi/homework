""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        # TODO 完成基础卷积结构：卷积->BN层->ReLU->卷积->BN层->ReLU,变量名可以自行设置，也可以延用已提供的变量名

        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO 推荐使用nn.Maxpool2d()，并调用上述以及定义的DoubleConv()完成下采样操作;变量名可以自行设置，也可以延用已提供的变量名
        # TODO 需要考虑添加注意力机制，注意力机制包括但不限于空间注意力机制、通道注意力机制。
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        # 通道注意力（直接内联实现）
        self.ch_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x=self.maxpool_conv(x)
        return x*self.ch_att(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # TODO 推荐使用nn.Upsample()，并调用上述以及定义的DoubleConv()完成上采样操作;变量名可以自行设置，也可以延用已提供的变量名
        # TODO 需要考虑添加注意力机制，注意力机制包括但不限于空间注意力机制、通道注意力机制。
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv_block = DoubleConv(out_channels * 2, out_channels)
        # 空间注意力（直接内联实现）
        self.sp_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if isinstance(self.up, nn.Upsample):
            x1 = self.conv(x1)

        # 尺寸对齐
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 拼接并应用空间注意力
        x = torch.cat([x2, x1], dim=1)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        return self.conv_block(x * self.sp_att(att))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    