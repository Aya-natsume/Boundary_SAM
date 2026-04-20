import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # 判别器里的残差块延续旧工程写法，激活换成 LeakyReLU，边缘更硬一点。
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, in_channels),
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 还是标准残差连接，没什么花活，稳定最重要。
        x = x + self.block(x)
        return self.relu(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # 这里做的是 PatchGAN 风格判别器，输入通常是 32 通道解码特征。
        self.model = nn.Sequential(
            # 第一层直接把空间分辨率减半，先抓局部真伪模式。
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 第二层继续下采样，同时扩大通道数。
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 第三层再压一层，感受野基本够看边界区域的结构差异了。
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 最后一层输出 patch-level 判别图，而不是单个标量。
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输出形状一般是 [B,1,h,w]，训练时通常接 MSELoss 或 BCE。
        return self.model(x)
