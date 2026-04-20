import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # 两层 3x3 卷积做残差映射，结构和旧工程保持一致，不然预训练权重会对不上。
        self.block = nn.Sequential(
            # 第一层卷积先提一下局部上下文，像是在边界附近多看一眼。
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # GroupNorm 对小 batch 更稳，这个项目里 batch size 一直都不算大。
            nn.GroupNorm(32, in_channels),
            # ReLU 保留非线性，简单，但够用。
            nn.ReLU(inplace=True),
            # 第二层卷积继续细化残差分支。
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # 归一化还是同一套配置，别乱动，权重兼容靠它撑着。
            nn.GroupNorm(32, in_channels),
        )
        # 主分支和残差分支相加之后再过一次激活，输出会更干净一点。
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 典型残差写法：输入先走一圈，再把自己接回来。
        x = x + self.block(x)
        return self.relu(x)


class Encoder(nn.Module):
    """
    Keep the module layout aligned with the Boundary reference implementation so
    old `.pth` checkpoints can be loaded directly after stripping `module.`.
    """

    def __init__(self, in_channels: int = 1, feature_channels: int = 256):
        super().__init__()
        # 这里故意锁死 256，不是在耍脾气，是为了和参考工程的权重键形状完全一致。
        if feature_channels != 256:
            raise ValueError("feature_channels must stay 256 to remain checkpoint-compatible")

        # 第一段编码：从单通道医学图像起步，先提浅层纹理，再降一次采样。
        self.block1 = nn.Sequential(
            # 输入一般是 [B,1,H,W]，先扩到 32 通道。
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=32),
            nn.ReLU(inplace=True),
            # 再来一层同尺度卷积，把局部结构再捋顺一点。
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=32),
            nn.ReLU(inplace=True),
            # stride=2 做第一次下采样，输出 64 通道。
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, num_channels=64),
            nn.ReLU(inplace=True),
        )
        # 第二段编码：继续提升语义层级，再往下压一层分辨率。
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, num_channels=128),
            nn.ReLU(inplace=True),
        )
        # 第三段编码：把特征推到 256 通道，后面残差块就在这个尺度上工作。
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, num_channels=256),
            nn.ReLU(inplace=True),
        )

        # 三个残差块沿用旧工程设计，用来补充高层语义表达。
        self.res1 = ResBlock(256)
        self.res2 = ResBlock(256)
        self.res3 = ResBlock(256)
        # InstanceNorm 放在末端，弱化域间风格差异，这种跨模态设定里挺常见。
        self.IN = nn.InstanceNorm2d(256, affine=False, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, use_multi_feature: bool = False):
        # out1/out2/out3 分别对应三个编码阶段，后面如果要做多尺度特征交互，可以直接拿走。
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)

        # 如果上层逻辑想自己处理多尺度特征，就在这里提前返回。
        if use_multi_feature:
            return out1, out2, out3

        # 默认路径继续走高层残差建模。
        x = self.res1(out3)
        x = self.res2(x)
        x = self.res3(x)
        # IN + ReLU 收个尾，让编码输出分布更稳定一些。
        x = self.IN(x)
        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, num_class: int, feature_channels: int = 256):
        super().__init__()
        # 解码器也锁死通道配置，不然旧 checkpoint 一样读不进来。
        if feature_channels != 256:
            raise ValueError("feature_channels must stay 256 to remain checkpoint-compatible")

        # 先在瓶颈层再做三次残差 refinement，跟参考工程完全同构。
        self.res1 = ResBlock(feature_channels)
        self.res2 = ResBlock(feature_channels)
        self.res3 = ResBlock(feature_channels)

        # 第一段上采样：256 -> 128，分辨率翻倍。
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(feature_channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
        )
        # 第二段上采样：128 -> 64，再翻一倍。
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        # 第三段上采样：64 -> 32，回到接近输入分辨率的语义图。
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(32, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(32, 32),
            nn.ReLU(inplace=True),
        )
        # 最后一层 3x3 卷积输出分类 logits，通道数等于类别数。
        self.block4 = nn.Sequential(
            nn.Conv2d(32, num_class, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, Seg_D2: bool = False) -> torch.Tensor:
        # 先过瓶颈残差块，把编码特征再打磨一遍。
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        # 再逐级上采样回到分割图大小。
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # 训练里如果要拿中间特征做对比学习或判别器，这里直接返回归一化特征。
        if Seg_D2:
            return F.normalize(x, dim=1)

        # 默认路径输出分割 logits，交给 CE 或 softmax 去处理。
        return self.block4(x)
