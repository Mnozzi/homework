""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # TODO 调用unet_parts.py中已完成的basic结构完成UNet网络整体结构的搭建，并添加相应的注释。
        # 编码器（下采样路径）
        self.inc = DoubleConv(n_channels, 64)  # 初始双卷积
        self.down1 = Down(64, 128)  # 下采样阶段1
        self.down2 = Down(128, 256)  # 下采样阶段2
        self.down3 = Down(256, 512)  # 下采样阶段3
        self.down4 = Down(512, 1024 if not bilinear else 512)  # 下采样阶段4

        # 解码器（上采样路径）
        self.up1 = Up(1024, 512, bilinear)  # 上采样阶段1
        self.up2 = Up(512, 256, bilinear)  # 上采样阶段2
        self.up3 = Up(256, 128, bilinear)  # 上采样阶段3
        self.up4 = Up(128, 64, bilinear)  # 上采样阶段4

        # 最终输出层
        self.outc = OutConv(64, n_classes)  # 输出卷积


    def forward(self, x):
        # TODO 调用unet_parts.py中已完成的basic结构完成UNet网络forward，并添加相应的注释。
        # 主要包括4次下采样和4次上采样，以及最后一层输出卷积。同时需要再U-Net中添加注意力机制，注意力机制包括但不限于空间注意力机制、通道注意力机制等。
        # 编码过程
        x1 = self.inc(x)  # 初始双卷积 [B,64,H,W]
        x2 = self.down1(x1)  # 下采样1 [B,128,H/2,W/2]
        x3 = self.down2(x2)  # 下采样2 [B,256,H/4,W/4]
        x4 = self.down3(x3)  # 下采样3 [B,512,H/8,W/8]
        x5 = self.down4(x4)  # 下采样4 [B,1024,H/16,W/16]（如果双线性则为512）

        # 解码过程（含跳跃连接）
        x = self.up1(x5, x4)  # 上采样1 + 拼接x4 [B,512,H/8,W/8]
        x = self.up2(x, x3)  # 上采样2 + 拼接x3 [B,256,H/4,W/4]
        x = self.up3(x, x2)  # 上采样3 + 拼接x2 [B,128,H/2,W/2]
        x = self.up4(x, x1)  # 上采样4 + 拼接x1 [B,64,H,W]

        # 最终输出
        output = self.outc(x)  # 输出预测 [B,n_classes,H,W]
        return output