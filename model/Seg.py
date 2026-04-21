"""兼容旧预训练权重的分割网络实现。"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_conv_norm_relu(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Sequential:
    """构建一个最基础的 Conv-GN-ReLU 模块。"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(32, out_channels),
        nn.ReLU(inplace=True),
    )


class ResBlock(nn.Module):
    """最简单的残差块，结构和旧模型保持一致。"""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        out = out + self.block(out)
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    """二维分割编码器。"""

    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            _make_conv_norm_relu(1, 32, stride=1),
            _make_conv_norm_relu(32, 32, stride=1),
            _make_conv_norm_relu(32, 64, stride=2),
        )
        self.block2 = nn.Sequential(
            _make_conv_norm_relu(64, 64, stride=1),
            _make_conv_norm_relu(64, 64, stride=1),
            _make_conv_norm_relu(64, 128, stride=2),
        )
        self.block3 = nn.Sequential(
            _make_conv_norm_relu(128, 128, stride=1),
            _make_conv_norm_relu(128, 128, stride=1),
            _make_conv_norm_relu(128, 256, stride=2),
        )
        self.res1 = ResBlock(256)
        self.res2 = ResBlock(256)
        self.res3 = ResBlock(256)
        self.IN = nn.InstanceNorm2d(256, affine=False, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        out: torch.Tensor,
        use_multi_feature: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out1 = self.block1(out)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        if use_multi_feature:
            return out1, out2, out3
        out = self.res1(out3)
        out = self.res2(out)
        out = self.res3(out)
        out = self.IN(out)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    """二维分割解码器。"""

    def __init__(self, num_class: int) -> None:
        super().__init__()
        self.res1 = ResBlock(256)
        self.res2 = ResBlock(256)
        self.res3 = ResBlock(256)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            _make_conv_norm_relu(128, 128, stride=1),
            _make_conv_norm_relu(128, 128, stride=1),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            _make_conv_norm_relu(64, 64, stride=1),
            _make_conv_norm_relu(64, 64, stride=1),
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, 32),
            nn.ReLU(inplace=True),
            _make_conv_norm_relu(32, 32, stride=1),
            _make_conv_norm_relu(32, 32, stride=1),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(32, num_class, kernel_size=3, padding=1),
        )

    def forward(self, out: torch.Tensor, Seg_D2: bool = False) -> torch.Tensor:
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        if Seg_D2:
            return F.normalize(out, dim=1)
        out = self.block4(out)
        return out


class BoundarySegmentationModel(nn.Module):
    """把编码器和解码器包起来，实验时调用会顺手很多。"""

    def __init__(self, num_class: int) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_class=num_class)

    def forward(self, image: torch.Tensor, return_feature: bool = False) -> torch.Tensor:
        feature = self.encoder(image)
        return self.decoder(feature, Seg_D2=return_feature)

    def load_pretrained(
        self,
        encoder_checkpoint: str | Path | None = None,
        decoder_checkpoint: str | Path | None = None,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> Dict[str, Tuple[str, ...]]:
        """分别加载编码器和解码器的预训练参数。"""
        load_report: Dict[str, Tuple[str, ...]] = {}
        if encoder_checkpoint is not None:
            incompatible = load_pretrained_weights(self.encoder, encoder_checkpoint, strict=strict, map_location=map_location)
            load_report["encoder_missing"] = tuple(incompatible.missing_keys)
            load_report["encoder_unexpected"] = tuple(incompatible.unexpected_keys)
        if decoder_checkpoint is not None:
            incompatible = load_pretrained_weights(self.decoder, decoder_checkpoint, strict=strict, map_location=map_location)
            load_report["decoder_missing"] = tuple(incompatible.missing_keys)
            load_report["decoder_unexpected"] = tuple(incompatible.unexpected_keys)
        return load_report


def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    """从不同格式的 checkpoint 里抽出真正的 state_dict。"""
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        return checkpoint["state_dict"]
    if isinstance(checkpoint, Mapping):
        return checkpoint
    raise TypeError("Checkpoint must be a mapping that contains tensor weights")


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    """去掉 DataParallel 留下的 module. 前缀。"""
    cleaned_state_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value
    return cleaned_state_dict


def load_pretrained_weights(
    module: nn.Module,
    checkpoint_path: str | Path,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> torch.nn.modules.module._IncompatibleKeys:
    """加载预训练权重，行为和旧训练脚本保持一致，但写法更集中。"""
    checkpoint = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location=map_location)
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_module_prefix(state_dict)
    incompatible = module.load_state_dict(state_dict, strict=strict)
    return incompatible

