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
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),  # 3x3 卷积是主力，不折腾奇怪配置。
        nn.GroupNorm(32, out_channels),  # 这里继续用 32 组，和旧权重严格对齐。
        nn.ReLU(inplace=True),  # ReLU 足够直接，也和参考实现一致。
    )


class ResBlock(nn.Module):
    """最简单的残差块，结构和旧模型保持一致。"""

    def __init__(self, in_channels: int) -> None:
        super().__init__()  # 父类初始化先走，别省这一步。
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # 第一层卷积先提特征。
            nn.GroupNorm(32, in_channels),  # GroupNorm 对小 batch 友好一些。
            nn.ReLU(inplace=True),  # 中间激活直接用 ReLU。
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # 第二层卷积负责补表达能力。
            nn.GroupNorm(32, in_channels),  # 归一化继续保持同配置，别乱改。
        )
        self.relu = nn.ReLU(inplace=True)  # 残差相加后的激活单独放出来，结构更清楚。

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        out = out + self.block(out)  # 标准残差连接，没必要故作神秘。
        out = self.relu(out)  # 相加后再激活，和旧实现一致。
        return out


class Encoder(nn.Module):
    """二维分割编码器。"""

    def __init__(self) -> None:
        super().__init__()  # 先初始化父类。
        self.block1 = nn.Sequential(
            _make_conv_norm_relu(1, 32, stride=1),  # 第一层把单通道输入提到 32 通道。
            _make_conv_norm_relu(32, 32, stride=1),  # 再做一层同分辨率卷积，先把局部纹理学稳。
            _make_conv_norm_relu(32, 64, stride=2),  # 这里第一次下采样，把特征尺度压下去。
        )
        self.block2 = nn.Sequential(
            _make_conv_norm_relu(64, 64, stride=1),  # 第二个编码块先做同尺度特征提炼。
            _make_conv_norm_relu(64, 64, stride=1),  # 再补一层，别太单薄。
            _make_conv_norm_relu(64, 128, stride=2),  # 第二次下采样，通道同步翻倍。
        )
        self.block3 = nn.Sequential(
            _make_conv_norm_relu(128, 128, stride=1),  # 第三个编码块继续堆表达。
            _make_conv_norm_relu(128, 128, stride=1),  # 结构重复一点，换来的是可预测性。
            _make_conv_norm_relu(128, 256, stride=2),  # 最后一次下采样，到 256 通道。
        )
        self.res1 = ResBlock(256)  # 编码末端接残差块，提升非线性能力。
        self.res2 = ResBlock(256)  # 第二个残差块继续稳住高层特征。
        self.res3 = ResBlock(256)  # 第三个残差块保持和旧实现一致。
        self.IN = nn.InstanceNorm2d(256, affine=False, track_running_stats=False)  # 这里保留旧成员名，方便严格加载旧权重。
        self.relu = nn.ReLU(inplace=True)  # 末端激活照旧。

    def forward(
        self,
        out: torch.Tensor,
        use_multi_feature: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out1 = self.block1(out)  # 第一阶段特征，分辨率下降 2 倍。
        out2 = self.block2(out1)  # 第二阶段特征，分辨率再降 2 倍。
        out3 = self.block3(out2)  # 第三阶段特征，得到最深层编码表示。
        if use_multi_feature:  # 如果训练策略需要中间层，就直接把三层都吐出来。
            return out1, out2, out3
        out = self.res1(out3)  # 残差块一层一层过，不玩花样。
        out = self.res2(out)  # 第二层残差继续提炼。
        out = self.res3(out)  # 第三层残差收尾。
        out = self.IN(out)  # 实例归一化一下，风格差异会收敛一点。
        out = self.relu(out)  # 最后再激活。
        return out


class Decoder(nn.Module):
    """二维分割解码器。"""

    def __init__(self, num_class: int) -> None:
        super().__init__()  # 父类初始化照例先走。
        self.res1 = ResBlock(256)  # 解码前先补三层残差，和参考实现对齐。
        self.res2 = ResBlock(256)  # 第二层残差。
        self.res3 = ResBlock(256)  # 第三层残差。
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 第一次上采样，尺寸翻倍。
            nn.GroupNorm(32, 128),  # 归一化继续保持 32 组。
            nn.ReLU(inplace=True),  # 激活还是 ReLU，足够了。
            _make_conv_norm_relu(128, 128, stride=1),  # 上采样后补两层卷积细化特征。
            _make_conv_norm_relu(128, 128, stride=1),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 第二次上采样。
            nn.GroupNorm(32, 64),  # 归一化参数继续沿用旧配置。
            nn.ReLU(inplace=True),
            _make_conv_norm_relu(64, 64, stride=1),  # 同分辨率细化。
            _make_conv_norm_relu(64, 64, stride=1),
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 第三次上采样，回到接近输入尺度。
            nn.GroupNorm(32, 32),
            nn.ReLU(inplace=True),
            _make_conv_norm_relu(32, 32, stride=1),  # 最后一段特征整理。
            _make_conv_norm_relu(32, 32, stride=1),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(32, num_class, kernel_size=3, padding=1),  # 最终分类头只做一层卷积，干净直接。
        )

    def forward(self, out: torch.Tensor, Seg_D2: bool = False) -> torch.Tensor:
        out = self.res1(out)  # 解码前的残差块一。
        out = self.res2(out)  # 解码前的残差块二。
        out = self.res3(out)  # 解码前的残差块三。
        out = self.block1(out)  # 上采样到 1/4 尺度。
        out = self.block2(out)  # 上采样到 1/2 尺度。
        out = self.block3(out)  # 上采样回原尺度。
        if Seg_D2:  # 如果要拿归一化特征做对比学习，就在这里返回。
            return F.normalize(out, dim=1)
        out = self.block4(out)  # 否则输出分割 logits。
        return out


class BoundarySegmentationModel(nn.Module):
    """把编码器和解码器包起来，实验时调用会顺手很多。"""

    def __init__(self, num_class: int) -> None:
        super().__init__()  # 父类初始化。
        self.encoder = Encoder()  # 编码器负责提深层语义。
        self.decoder = Decoder(num_class=num_class)  # 解码器负责恢复分割图。

    def forward(self, image: torch.Tensor, return_feature: bool = False) -> torch.Tensor:
        feature = self.encoder(image)  # 先编码，没什么悬念。
        return self.decoder(feature, Seg_D2=return_feature)  # 按需返回特征或 logits。

    def load_pretrained(
        self,
        encoder_checkpoint: str | Path | None = None,
        decoder_checkpoint: str | Path | None = None,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> Dict[str, Tuple[str, ...]]:
        """分别加载编码器和解码器的预训练参数。"""
        load_report: Dict[str, Tuple[str, ...]] = {}  # 把每次加载的缺失项和多余项都记下来，排查会省事。
        if encoder_checkpoint is not None:  # 给了编码器权重就加载。
            incompatible = load_pretrained_weights(self.encoder, encoder_checkpoint, strict=strict, map_location=map_location)
            load_report["encoder_missing"] = tuple(incompatible.missing_keys)  # 缺哪些键，明明白白记下来。
            load_report["encoder_unexpected"] = tuple(incompatible.unexpected_keys)  # 多哪些键，也记下来。
        if decoder_checkpoint is not None:  # 解码器同理。
            incompatible = load_pretrained_weights(self.decoder, decoder_checkpoint, strict=strict, map_location=map_location)
            load_report["decoder_missing"] = tuple(incompatible.missing_keys)
            load_report["decoder_unexpected"] = tuple(incompatible.unexpected_keys)
        return load_report


def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    """从不同格式的 checkpoint 里抽出真正的 state_dict。"""
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        return checkpoint["state_dict"]  # 有些训练脚本喜欢把权重包在 state_dict 下面，顺手拆掉。
    if isinstance(checkpoint, Mapping):
        return checkpoint  # 旧项目里多数情况直接就是参数字典。
    raise TypeError("Checkpoint must be a mapping that contains tensor weights")  # 不是字典就别硬装了。


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    """去掉 DataParallel 留下的 module. 前缀。"""
    cleaned_state_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()  # 用有序字典保存，阅读和调试都更直观。
    for key, value in state_dict.items():  # 逐个键处理，逻辑最透明。
        new_key = key[7:] if key.startswith("module.") else key  # 前缀存在就切掉，不存在就保持原样。
        cleaned_state_dict[new_key] = value
    return cleaned_state_dict


def load_pretrained_weights(
    module: nn.Module,
    checkpoint_path: str | Path,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> torch.nn.modules.module._IncompatibleKeys:
    """加载预训练权重，行为和旧训练脚本保持一致，但写法更集中。"""
    checkpoint = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location=map_location)  # 先把 checkpoint 读进来。
    state_dict = _extract_state_dict(checkpoint)  # 抽出真正的参数字典。
    state_dict = _strip_module_prefix(state_dict)  # 把 module. 前缀统一去掉。
    incompatible = module.load_state_dict(state_dict, strict=strict)  # 最后交给 PyTorch 正式加载。
    return incompatible

