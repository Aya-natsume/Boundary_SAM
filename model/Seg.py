import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.block(x)
        return self.relu(x)


class Encoder(nn.Module):
    """
    Keep the module layout aligned with the Boundary reference implementation so
    old `.pth` checkpoints can be loaded directly after stripping `module.`.
    """

    def __init__(self, in_channels: int = 1, feature_channels: int = 256):
        super().__init__()
        if feature_channels != 256:
            raise ValueError("feature_channels must stay 256 to remain checkpoint-compatible")

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(32, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, num_channels=64),
            nn.ReLU(inplace=True),
        )
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

        self.res1 = ResBlock(256)
        self.res2 = ResBlock(256)
        self.res3 = ResBlock(256)
        self.IN = nn.InstanceNorm2d(256, affine=False, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, use_multi_feature: bool = False):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)

        if use_multi_feature:
            return out1, out2, out3

        x = self.res1(out3)
        x = self.res2(x)
        x = self.res3(x)
        x = self.IN(x)
        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, num_class: int, feature_channels: int = 256):
        super().__init__()
        if feature_channels != 256:
            raise ValueError("feature_channels must stay 256 to remain checkpoint-compatible")

        self.res1 = ResBlock(feature_channels)
        self.res2 = ResBlock(feature_channels)
        self.res3 = ResBlock(feature_channels)

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
        self.block4 = nn.Sequential(
            nn.Conv2d(32, num_class, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, Seg_D2: bool = False) -> torch.Tensor:
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        if Seg_D2:
            return F.normalize(x, dim=1)

        return self.block4(x)

