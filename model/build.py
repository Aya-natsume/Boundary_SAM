from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

import torch
import torch.nn as nn

from .ID import PatchDiscriminator
from .Seg import Decoder, Encoder


def remove_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, MutableMapping):
        for key in ("state_dict", "model", "module", "teacher", "student"):
            value = checkpoint.get(key)
            if isinstance(value, MutableMapping):
                return value
    return checkpoint


def load_checkpoint_state_dict(checkpoint_path: str, map_location: Optional[str] = "cpu") -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _unwrap_state_dict(checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Unsupported checkpoint format for {checkpoint_path}")
    return remove_module_prefix(state_dict)


def load_pretrained_weights(
    module: nn.Module,
    checkpoint_path: str,
    strict: bool = True,
    map_location: Optional[str] = "cpu",
):
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
    models = {
        "SE_target": Encoder(),
        "SE_source": Encoder(),
        "SD": Decoder(num_class=num_classes),
        "FD": PatchDiscriminator(in_channels=discriminator_in_channels),
    }

    if encoder_target_ckpt is not None:
        load_pretrained_weights(models["SE_target"], encoder_target_ckpt, strict=strict)
    if encoder_source_ckpt is not None:
        load_pretrained_weights(models["SE_source"], encoder_source_ckpt, strict=strict)
    if decoder_ckpt is not None:
        load_pretrained_weights(models["SD"], decoder_ckpt, strict=strict)
    if discriminator_ckpt is not None:
        load_pretrained_weights(models["FD"], discriminator_ckpt, strict=strict)

    if device is not None:
        models = {name: model.to(device) for name, model in models.items()}

    return models

