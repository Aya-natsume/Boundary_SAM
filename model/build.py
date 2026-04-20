from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

import torch
import torch.nn as nn

from .ID import PatchDiscriminator
from .Seg import Decoder, Encoder


def remove_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # 旧项目里不少权重是 DataParallel / EMA 存出来的，`module.` 前缀得先清掉。
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def _unwrap_state_dict(checkpoint):
    # 有些 checkpoint 外面还包了一层 dict，这里顺手把常见包装拆开。
    if isinstance(checkpoint, MutableMapping):
        for key in ("state_dict", "model", "module", "teacher", "student"):
            value = checkpoint.get(key)
            if isinstance(value, MutableMapping):
                return value
    return checkpoint


def load_checkpoint_state_dict(checkpoint_path: str, map_location: Optional[str] = "cpu") -> Dict[str, torch.Tensor]:
    # 默认先在 CPU 上读，省得一上来就把显存挤得不耐烦。
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    # 把外层包装拆掉，尽量统一成标准 state_dict。
    state_dict = _unwrap_state_dict(checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Unsupported checkpoint format for {checkpoint_path}")
    # 最后把多卡训练遗留的前缀处理掉，后面 load_state_dict 就会顺很多。
    return remove_module_prefix(state_dict)


def load_pretrained_weights(
    module: nn.Module,
    checkpoint_path: str,
    strict: bool = True,
    map_location: Optional[str] = "cpu",
):
    # 单个模块的统一加载入口，训练脚本里就不用一遍遍重复写样板代码了。
    state_dict = load_checkpoint_state_dict(checkpoint_path, map_location=map_location)
    return module.load_state_dict(state_dict, strict=strict)


def build_segmentation_models(
    num_classes: int,
    discriminator_in_channels: int = 32,
    encoder_target_ckpt: Optional[str] = None,
    encoder_source_ckpt: Optional[str] = None,
    decoder_ckpt: Optional[str] = None,
    discriminator_ckpt: Optional[str] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
):
    # 这里把训练阶段会用到的四个核心模块一次性建好，省得外面再拼。
    models = {
        # 目标域编码器，对应原工程里的 SE_target。
        "SE_target": Encoder(),
        # 源域编码器，对应原工程里的 SE_source。
        "SE_source": Encoder(),
        # 分割解码器，对应原工程里的 SD。
        "SD": Decoder(num_class=num_classes),
        # 特征判别器，对应原工程里的 FD。
        "FD": PatchDiscriminator(in_channels=discriminator_in_channels),
    }

    # 下面这些加载都是可选的，不传路径就保持随机初始化。
    if encoder_target_ckpt is not None:
        load_pretrained_weights(models["SE_target"], encoder_target_ckpt, strict=strict)
    if encoder_source_ckpt is not None:
        load_pretrained_weights(models["SE_source"], encoder_source_ckpt, strict=strict)
    if decoder_ckpt is not None:
        load_pretrained_weights(models["SD"], decoder_ckpt, strict=strict)
    if discriminator_ckpt is not None:
        load_pretrained_weights(models["FD"], discriminator_ckpt, strict=strict)

    # 如果外面已经决定了设备，这里顺手把模型搬过去，别让训练脚本再重复搬一次。
    if device is not None:
        models = {name: model.to(device) for name, model in models.items()}

    return models
