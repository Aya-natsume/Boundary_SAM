from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


BoundaryKey = Tuple[int, int]


def canonicalize_boundary_key(a: int, b: int, ordered: bool = False) -> BoundaryKey:
    """
    将边界类别对统一规范化为稳定 key，避免同一条边界被重复访问。

    规范规则:
        1. `ordered=True` 时，严格保留输入顺序，返回 `(a, b)`
        2. `ordered=False` 时，器官-器官边界统一为 `(min(a, b), max(a, b))`
        3. `ordered=False` 且存在背景类别 0 时，统一写成 `(foreground, 0)`

    参数:
        a: int
            第一个类别编号。
        b: int
            第二个类别编号。
        ordered: bool
            是否保留有序边界表示。

    返回:
        boundary_key: Tuple[int, int]
            规范化后的边界类别 key。
    """
    a = int(a)
    b = int(b)
    if a == b:
        raise ValueError(f"边界类别对中的两个类别不能相同，当前为 {(a, b)}")

    if ordered:
        return a, b

    if a == 0 or b == 0:
        foreground_class = b if a == 0 else a
        return int(foreground_class), 0

    return tuple(sorted((a, b)))


def _validate_kernel_size(kernel_size: int) -> int:
    """
    校验边界带提取时使用的形态学核大小是否合法。

    参数:
        kernel_size: int
            邻域核大小，必须为正奇数。

    返回:
        kernel_size: int
            通过校验后的核大小。
    """
    if not isinstance(kernel_size, int):
        raise TypeError(f"kernel_size 必须是 int，当前类型为 {type(kernel_size)}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size 必须是正奇数，当前为 {kernel_size}")
    return kernel_size


def _prepare_binary_mask(mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    将输入二值 mask 统一整理成 `(B, 1, H, W)` 形式，便于后续使用池化模拟形态学操作。

    参数:
        mask: torch.Tensor
            支持 `(H, W)`、`(B, H, W)`、`(B, 1, H, W)` 三种输入形状。

    返回:
        mask_4d: torch.Tensor
            统一后的 4 维二值张量，形状为 `(B, 1, H, W)`。
        original_ndim: int
            原始维度，用于在函数末尾恢复形状。
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask 必须是 torch.Tensor")

    original_ndim = mask.ndim
    if original_ndim == 2:
        mask_4d = mask.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 3:
        mask_4d = mask.unsqueeze(1)
    elif original_ndim == 4:
        if mask.shape[1] != 1:
            raise ValueError(f"4 维 mask 的通道数必须为 1，当前 shape={tuple(mask.shape)}")
        mask_4d = mask
    else:
        raise ValueError(f"mask 必须是 2/3/4 维张量，当前 ndim={mask.ndim}")

    # 这里统一成 0/1 float，后面 max-pool 和 min-pool 替代实现就会很顺手。
    mask_4d = (mask_4d > 0).to(dtype=torch.float32)
    return mask_4d, original_ndim


def _restore_binary_mask(mask_4d: torch.Tensor, original_ndim: int) -> torch.Tensor:
    """
    将 `(B, 1, H, W)` 形式的二值张量恢复成调用者原始维度。

    参数:
        mask_4d: torch.Tensor
            形状为 `(B, 1, H, W)` 的张量。
        original_ndim: int
            调用前输入张量的维度。

    返回:
        restored_mask: torch.Tensor
            恢复维度后的张量。
    """
    if original_ndim == 2:
        return mask_4d[0, 0]
    if original_ndim == 3:
        return mask_4d[:, 0]
    if original_ndim == 4:
        return mask_4d
    raise ValueError(f"不支持的 original_ndim={original_ndim}")


def _ensure_feature_map_is_bchw(feature_map: torch.Tensor) -> torch.Tensor:
    """
    将输入特征图整理成 `(B, C, H, W)` 形式。

    参数:
        feature_map: torch.Tensor
            支持 `(C, H, W)` 或 `(B, C, H, W)`。

    返回:
        feature_map_bchw: torch.Tensor
            统一后的特征图张量。
    """
    if not isinstance(feature_map, torch.Tensor):
        raise TypeError("feature_map 必须是 torch.Tensor")

    # 单张图 `[C,H,W]` 时补出 batch 维，后面所有函数就都能走统一路径。
    if feature_map.ndim == 3:
        feature_map = feature_map.unsqueeze(0)
    elif feature_map.ndim != 4:
        raise ValueError(f"feature_map 必须是 3/4 维张量，当前 shape={tuple(feature_map.shape)}")

    return feature_map


def _ensure_prob_map_is_bkhw(prob_map: torch.Tensor) -> torch.Tensor:
    """
    将概率图整理成 `(B, K, H, W)` 形式。

    参数:
        prob_map: torch.Tensor
            支持 `(K, H, W)` 或 `(B, K, H, W)`。

    返回:
        prob_map_bkhw: torch.Tensor
            统一后的概率图张量。
    """
    if not isinstance(prob_map, torch.Tensor):
        raise TypeError("prob_map 必须是 torch.Tensor")

    if prob_map.ndim == 3:
        prob_map = prob_map.unsqueeze(0)
    elif prob_map.ndim != 4:
        raise ValueError(f"prob_map 必须是 3/4 维张量，当前 shape={tuple(prob_map.shape)}")

    return prob_map.to(dtype=torch.float32)


def _ensure_label_is_bhw(seg_label: torch.Tensor) -> torch.Tensor:
    """
    将标签图统一整理成 `(B, H, W)` 形式。

    参数:
        seg_label: torch.Tensor
            支持 `(H, W)` 或 `(B, H, W)`。

    返回:
        seg_label_bhw: torch.Tensor
            统一后的 long 标签张量。
    """
    if not isinstance(seg_label, torch.Tensor):
        raise TypeError("seg_label 必须是 torch.Tensor")

    if seg_label.ndim == 2:
        seg_label = seg_label.unsqueeze(0)
    elif seg_label.ndim != 3:
        raise ValueError(f"seg_label 必须是 2/3 维张量，当前 shape={tuple(seg_label.shape)}")

    return seg_label.long()


def _ensure_map_is_bhw(spatial_map: torch.Tensor) -> torch.Tensor:
    """
    将任意空间图整理成 `(B, H, W)` 形式，并保留原始 dtype。

    参数:
        spatial_map: torch.Tensor
            支持 `(H, W)` 或 `(B, H, W)`。

    返回:
        spatial_map_bhw: torch.Tensor
            统一后的空间图张量。
    """
    if not isinstance(spatial_map, torch.Tensor):
        raise TypeError("spatial_map 必须是 torch.Tensor")

    if spatial_map.ndim == 2:
        spatial_map = spatial_map.unsqueeze(0)
    elif spatial_map.ndim != 3:
        raise ValueError(f"spatial_map 必须是 2/3 维张量，当前 shape={tuple(spatial_map.shape)}")

    return spatial_map


def extract_morph_boundary(
    mask: torch.Tensor,
    kernel_size: int = 3,
    mode: str = "inner",
) -> torch.Tensor:
    """
    使用 PyTorch 形态学近似操作快速提取二值 mask 的边界带。

    支持模式:
        1. `inner`: `mask - erode(mask)`，返回目标内部贴边的一圈
        2. `gradient`: `dilate(mask) - erode(mask)`，返回更宽的形态学梯度边界

    参数:
        mask: torch.Tensor
            单类二值 mask，支持 `(H, W)`、`(B, H, W)`、`(B, 1, H, W)`。
        kernel_size: int
            形态学核大小，必须为正奇数。
        mode: str
            边界提取模式，支持 `"inner"` 或 `"gradient"`。

    返回:
        boundary_mask: torch.Tensor
            与输入维度对齐的边界带 mask，数值类型为 bool。
    """
    kernel_size = _validate_kernel_size(kernel_size)
    mask_4d, original_ndim = _prepare_binary_mask(mask)
    padding = kernel_size // 2

    # 膨胀直接用 max_pool2d 近似实现，速度快，也不用引额外依赖。
    dilated = F.max_pool2d(mask_4d, kernel_size=kernel_size, stride=1, padding=padding)
    # 腐蚀等价于 min-pool，这里借助 `-max_pool2d(-x)` 实现。
    eroded = -F.max_pool2d(-mask_4d, kernel_size=kernel_size, stride=1, padding=padding)

    if mode == "inner":
        # inner 模式只保留器官内部靠边的一圈，更符合“粗边界带约束”的使用习惯。
        boundary = (mask_4d - eroded) > 0
    elif mode == "gradient":
        # gradient 会同时覆盖内外两侧，后续若想让边界带更宽可以切到这个模式。
        boundary = (dilated - eroded) > 0
    else:
        raise ValueError(f"mode 只支持 'inner' 或 'gradient'，当前为 {mode}")

    return _restore_binary_mask(boundary, original_ndim)


def extract_top1_top2(
    prob_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从粗分割概率图中提取每个位置的 top1 / top2 概率与类别。

    参数:
        prob_map: torch.Tensor
            概率图，形状为 `(B, K, H, W)` 或 `(K, H, W)`。

    返回:
        top1_prob: torch.Tensor
            top1 概率图，形状为 `(B, H, W)`。
        top2_prob: torch.Tensor
            top2 概率图，形状为 `(B, H, W)`。
        top1_cls: torch.Tensor
            top1 类别图，形状为 `(B, H, W)`。
        top2_cls: torch.Tensor
            top2 类别图，形状为 `(B, H, W)`。
    """
    prob_map = _ensure_prob_map_is_bkhw(prob_map)
    if prob_map.shape[1] < 2:
        raise ValueError(f"prob_map 的类别数必须至少为 2，当前 K={prob_map.shape[1]}")

    # 这里直接在类别维做 topk，避免手搓排序，速度和可读性都更稳一点。
    top_values, top_indices = torch.topk(prob_map, k=2, dim=1, largest=True, sorted=True)
    top1_prob = top_values[:, 0]
    top2_prob = top_values[:, 1]
    top1_cls = top_indices[:, 0].long()
    top2_cls = top_indices[:, 1].long()
    return top1_prob, top2_prob, top1_cls, top2_cls


def build_coarse_boundary_band(
    pred_mask: torch.Tensor,
    organ_id: int,
    kernel_size: int = 3,
    mode: str = "inner",
) -> torch.Tensor:
    """
    基于粗分割类别图，提取指定器官的粗边界带。

    参数:
        pred_mask: torch.Tensor
            粗分割类别图，形状为 `(B, H, W)` 或 `(H, W)`。
        organ_id: int
            当前器官类别编号。
        kernel_size: int
            形态学核大小，必须为正奇数。
        mode: str
            边界模式，默认 `"inner"`。

    返回:
        boundary_band: torch.Tensor
            指定器官的粗边界带，形状为 `(B, H, W)`，dtype 为 bool。
    """
    pred_mask = _ensure_label_is_bhw(pred_mask)
    organ_mask = pred_mask.eq(int(organ_id))

    # 这里延续源域阶段的形态学实现风格，用同一套 inner / gradient 约定。
    boundary_band = extract_morph_boundary(
        mask=organ_mask,
        kernel_size=kernel_size,
        mode=mode,
    ).bool()
    return boundary_band


def compute_uncertainty_from_margin(
    top1_prob: torch.Tensor,
    top2_prob: torch.Tensor,
    eps: float = 1e-6,
    clamp: bool = True,
) -> torch.Tensor:
    """
    使用 top1-top2 margin 构造轻量不确定性图。

    定义:
        uncertainty = 1 - (top1_prob - top2_prob)

    参数:
        top1_prob: torch.Tensor
            top1 概率图，形状为 `(B, H, W)`。
        top2_prob: torch.Tensor
            top2 概率图，形状为 `(B, H, W)`。
        eps: float
            数值稳定项。当前公式里不强依赖，但保留接口方便后续扩展。
        clamp: bool
            是否将结果裁剪到 `[0, 1]`。

    返回:
        uncertainty_map: torch.Tensor
            不确定性图，形状为 `(B, H, W)`。
    """
    if top1_prob.shape != top2_prob.shape:
        raise ValueError(
            f"top1_prob 与 top2_prob 的形状必须一致，当前为 {tuple(top1_prob.shape)} 和 {tuple(top2_prob.shape)}"
        )

    margin = top1_prob.to(dtype=torch.float32) - top2_prob.to(dtype=torch.float32)
    uncertainty_map = 1.0 - margin + float(eps) * 0.0
    if clamp:
        uncertainty_map = uncertainty_map.clamp(min=0.0, max=1.0)
    return uncertainty_map


def build_organ_relevance_mask(
    top1_cls: torch.Tensor,
    top2_cls: torch.Tensor,
    organ_id: int,
) -> torch.Tensor:
    """
    构造指定器官的相关性项 `M_A(i)`。

    定义:
        `M_A(i) = 1 if organ_id in {top1_cls(i), top2_cls(i)} else 0`

    参数:
        top1_cls: torch.Tensor
            top1 类别图，形状为 `(B, H, W)`。
        top2_cls: torch.Tensor
            top2 类别图，形状为 `(B, H, W)`。
        organ_id: int
            当前器官类别编号。

    返回:
        relevance_mask: torch.Tensor
            器官相关性 mask，形状为 `(B, H, W)`，dtype 为 float32。
    """
    if top1_cls.shape != top2_cls.shape:
        raise ValueError(
            f"top1_cls 与 top2_cls 的形状必须一致，当前为 {tuple(top1_cls.shape)} 和 {tuple(top2_cls.shape)}"
        )

    relevance_mask = top1_cls.eq(int(organ_id)) | top2_cls.eq(int(organ_id))
    return relevance_mask.to(dtype=torch.float32)


def compute_prototype_similarity_for_organ(
    feature_map: torch.Tensor,
    top1_cls: torch.Tensor,
    top2_cls: torch.Tensor,
    organ_id: int,
    prototype_library: Mapping[BoundaryKey, torch.Tensor],
    ordered: bool = False,
    normalize_feature: bool = True,
    missing_proto_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    为指定器官计算基于边界 prototype 的相似度图。

    计算逻辑:
        1. 仅在 `organ_id in {top1_cls, top2_cls}` 的位置上参与计算
        2. 对这些位置，定义另一个候选类别为 `b(i)`
        3. 仅访问 `(organ_id, b(i))` 对应的 prototype
        4. 计算当前位置特征与该 prototype 的余弦相似度

    参数:
        feature_map: torch.Tensor
            特征图，形状为 `(B, C, H, W)` 或 `(C, H, W)`。
        top1_cls: torch.Tensor
            top1 类别图，形状为 `(B, H, W)`。
        top2_cls: torch.Tensor
            top2 类别图，形状为 `(B, H, W)`。
        organ_id: int
            当前器官类别编号。
        prototype_library: Mapping[BoundaryKey, torch.Tensor]
            源域细粒度边界原型库，value 形状为 `(C,)`。
        ordered: bool
            是否保留有序边界表示。
        normalize_feature: bool
            是否先对 feature_map 做 L2 normalize。
        missing_proto_value: float
            若某个 prototype 缺失，则对应位置的相似度填充值。

    返回:
        proto_sim_map: torch.Tensor
            prototype 相似度图，形状为 `(B, H, W)`。
        conflict_cls_map: torch.Tensor
            每个位置对应的冲突类别图，形状为 `(B, H, W)`。
            若当前位置与 organ_id 无关，则填为 `-1`。
    """
    feature_map = _ensure_feature_map_is_bchw(feature_map).to(dtype=torch.float32)
    top1_cls = _ensure_label_is_bhw(top1_cls)
    top2_cls = _ensure_label_is_bhw(top2_cls)

    if top1_cls.shape != top2_cls.shape:
        raise ValueError(
            f"top1_cls 与 top2_cls 的形状必须一致，当前为 {tuple(top1_cls.shape)} 和 {tuple(top2_cls.shape)}"
        )
    if feature_map.shape[0] != top1_cls.shape[0] or feature_map.shape[-2:] != top1_cls.shape[-2:]:
        raise ValueError(
            f"feature_map 与 top1/top2 类别图的 batch 或空间尺寸不一致，"
            f"feature_map.shape={tuple(feature_map.shape)}, top_cls.shape={tuple(top1_cls.shape)}"
        )

    batch_size, feature_dim, height, width = feature_map.shape
    feature_map_norm = (
        F.normalize(feature_map, p=2, dim=1, eps=1e-12)
        if normalize_feature
        else feature_map
    )

    relevance_mask_bool = top1_cls.eq(int(organ_id)) | top2_cls.eq(int(organ_id))
    # 对相关位置，冲突类别就是 top1 / top2 里那个“不是 organ_id 的类别”。
    conflict_cls_map = torch.where(top1_cls.eq(int(organ_id)), top2_cls, top1_cls)
    conflict_cls_map = torch.where(
        relevance_mask_bool,
        conflict_cls_map,
        torch.full_like(conflict_cls_map, fill_value=-1),
    )

    proto_sim_map = feature_map.new_zeros((batch_size, height, width), dtype=torch.float32)
    unique_conflict_classes = torch.unique(conflict_cls_map[relevance_mask_bool])

    # 这里只在“候选冲突类别”层面做循环，不对像素做 Python 大循环。
    for conflict_class_tensor in unique_conflict_classes.tolist():
        conflict_class = int(conflict_class_tensor)
        if conflict_class < 0:
            continue

        pair_mask = relevance_mask_bool & conflict_cls_map.eq(conflict_class)
        pair_key = canonicalize_boundary_key(organ_id, conflict_class, ordered=ordered)
        prototype = prototype_library.get(pair_key, None)

        if prototype is None:
            # prototype 缺失时只给当前 pair 的位置写 missing 值，其它位置保持自己的默认值。
            proto_sim_map[pair_mask] = float(missing_proto_value)
            continue

        prototype = prototype.to(device=feature_map.device, dtype=torch.float32)
        if prototype.ndim != 1 or prototype.shape[0] != feature_dim:
            raise ValueError(
                f"prototype_library[{pair_key}] 的形状必须是 {(feature_dim,)}, 当前为 {tuple(prototype.shape)}"
            )

        # 余弦相似度要求 prototype 也先做 L2 normalize，别把量纲误差带进来。
        prototype = F.normalize(prototype.unsqueeze(0), p=2, dim=1, eps=1e-12).squeeze(0)
        sim_map_this_pair = (feature_map_norm * prototype.view(1, feature_dim, 1, 1)).sum(dim=1)
        proto_sim_map[pair_mask] = sim_map_this_pair[pair_mask]

    return proto_sim_map, conflict_cls_map.long()


def _compute_boundary_prompt_score_for_organ_from_top12(
    feature_map: torch.Tensor,
    pred_mask: torch.Tensor,
    top1_prob: torch.Tensor,
    top2_prob: torch.Tensor,
    top1_cls: torch.Tensor,
    top2_cls: torch.Tensor,
    organ_id: int,
    prototype_library: Mapping[BoundaryKey, torch.Tensor],
    boundary_kernel: int = 3,
    ordered: bool = False,
    missing_proto_value: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    基于已经提取好的 top1/top2 结果，为指定器官计算 boundary prompt score。

    参数:
        feature_map: torch.Tensor
            特征图，形状为 `(B, C, H, W)`。
        pred_mask: torch.Tensor
            粗分割类别图，形状为 `(B, H, W)`。
        top1_prob: torch.Tensor
            top1 概率图，形状为 `(B, H, W)`。
        top2_prob: torch.Tensor
            top2 概率图，形状为 `(B, H, W)`。
        top1_cls: torch.Tensor
            top1 类别图，形状为 `(B, H, W)`。
        top2_cls: torch.Tensor
            top2 类别图，形状为 `(B, H, W)`。
        organ_id: int
            当前器官类别编号。
        prototype_library: Mapping[BoundaryKey, torch.Tensor]
            细粒度边界单原型库。
        boundary_kernel: int
            粗边界带提取核大小。
        ordered: bool
            是否保留有序边界表示。
        missing_proto_value: float
            缺失 prototype 时的相似度填充值。

    返回:
        score_map: torch.Tensor
            当前器官的 boundary prompt score map，形状为 `(B, H, W)`。
        aux_dict: Dict[str, torch.Tensor]
            中间结果字典，便于后续检查。
    """
    relevance_mask = build_organ_relevance_mask(
        top1_cls=top1_cls,
        top2_cls=top2_cls,
        organ_id=organ_id,
    )
    boundary_band = build_coarse_boundary_band(
        pred_mask=pred_mask,
        organ_id=organ_id,
        kernel_size=boundary_kernel,
        mode="inner",
    ).to(dtype=torch.float32)
    uncertainty_map = compute_uncertainty_from_margin(
        top1_prob=top1_prob,
        top2_prob=top2_prob,
        clamp=True,
    )
    proto_sim_map, conflict_cls_map = compute_prototype_similarity_for_organ(
        feature_map=feature_map,
        top1_cls=top1_cls,
        top2_cls=top2_cls,
        organ_id=organ_id,
        prototype_library=prototype_library,
        ordered=ordered,
        normalize_feature=True,
        missing_proto_value=missing_proto_value,
    )

    # 当前阶段先严格按题目要求做乘法门控，不引入任何可学习融合权重。
    score_map = relevance_mask * boundary_band * uncertainty_map * proto_sim_map
    aux_dict = {
        "relevance_mask": relevance_mask,
        "boundary_band": boundary_band,
        "uncertainty_map": uncertainty_map,
        "proto_sim_map": proto_sim_map,
        "conflict_cls_map": conflict_cls_map,
        "top1_cls": top1_cls,
        "top2_cls": top2_cls,
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
    }
    return score_map, aux_dict


def compute_boundary_prompt_score_for_organ(
    feature_map: torch.Tensor,
    prob_map: torch.Tensor,
    pred_mask: torch.Tensor,
    organ_id: int,
    prototype_library: Mapping[BoundaryKey, torch.Tensor],
    boundary_kernel: int = 3,
    ordered: bool = False,
    missing_proto_value: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    为指定器官计算 boundary prompt score map。

    处理流程:
        1. 提取 top1 / top2 概率与类别
        2. 计算 uncertainty map
        3. 构造 organ relevance mask
        4. 构造 coarse boundary band
        5. 计算 prototype similarity map
        6. 用乘法门控融合成最终 score

    参数:
        feature_map: torch.Tensor
            特征图，形状为 `(B, C, H, W)`。
        prob_map: torch.Tensor
            粗分割概率图，形状为 `(B, K, H, W)`。
        pred_mask: torch.Tensor
            粗分割类别图，形状为 `(B, H, W)`。
        organ_id: int
            当前器官类别编号。
        prototype_library: Mapping[BoundaryKey, torch.Tensor]
            细粒度边界单原型库。
        boundary_kernel: int
            粗边界带核大小。
        ordered: bool
            是否保留有序边界表示。
        missing_proto_value: float
            若 prototype 缺失，相似度填充值。

    返回:
        score_map: torch.Tensor
            当前器官的 score map，形状为 `(B, H, W)`。
        aux_dict: Dict[str, torch.Tensor]
            中间结果字典。
    """
    feature_map = _ensure_feature_map_is_bchw(feature_map)
    prob_map = _ensure_prob_map_is_bkhw(prob_map)
    pred_mask = _ensure_label_is_bhw(pred_mask)

    top1_prob, top2_prob, top1_cls, top2_cls = extract_top1_top2(prob_map)
    return _compute_boundary_prompt_score_for_organ_from_top12(
        feature_map=feature_map,
        pred_mask=pred_mask,
        top1_prob=top1_prob,
        top2_prob=top2_prob,
        top1_cls=top1_cls,
        top2_cls=top2_cls,
        organ_id=organ_id,
        prototype_library=prototype_library,
        boundary_kernel=boundary_kernel,
        ordered=ordered,
        missing_proto_value=missing_proto_value,
    )


def select_topk_prompt_seeds(
    score_map: torch.Tensor,
    conflict_cls_map: torch.Tensor,
    topk: int = 10,
    min_score: float = 0.0,
    min_distance: int = 5,
) -> List[Dict[str, Union[int, float]]]:
    """
    从 score map 中按图像分别选取 top-k 高分边界锚点。

    去冗余策略:
        这里使用 Chebyshev 距离，
        即 `distance = max(abs(dy), abs(dx))`。
        如果新点与已选点的 Chebyshev 距离小于 `min_distance`，则跳过该点。

    参数:
        score_map: torch.Tensor
            score map，形状为 `(B, H, W)` 或 `(H, W)`。
        conflict_cls_map: torch.Tensor
            冲突类别图，形状为 `(B, H, W)` 或 `(H, W)`。
        topk: int
            每张图最多保留多少个点。
        min_score: float
            只保留分数大于该阈值的候选点。
        min_distance: int
            选点去冗余的最小 Chebyshev 距离阈值。

    返回:
        prompt_seed_list: List[Dict[str, Union[int, float]]]
            所有选中的 prompt seeds 列表。
    """
    score_map = _ensure_map_is_bhw(score_map.to(dtype=torch.float32))
    conflict_cls_map = _ensure_label_is_bhw(conflict_cls_map)

    if score_map.shape != conflict_cls_map.shape:
        raise ValueError(
            f"score_map 与 conflict_cls_map 的形状必须一致，当前为 {tuple(score_map.shape)} 和 {tuple(conflict_cls_map.shape)}"
        )
    if topk <= 0:
        return []

    prompt_seed_list: List[Dict[str, Union[int, float]]] = []
    batch_size = score_map.shape[0]

    for batch_idx in range(batch_size):
        # 先过滤掉低分点，这样后面的排序和去冗余都更省事。
        candidate_mask = score_map[batch_idx] > float(min_score)
        if not bool(candidate_mask.any()):
            continue

        candidate_coords = torch.nonzero(candidate_mask, as_tuple=False)
        candidate_scores = score_map[batch_idx][candidate_mask]
        sort_indices = torch.argsort(candidate_scores, descending=True)

        selected_points: List[Tuple[int, int]] = []
        for sorted_idx in sort_indices.tolist():
            y = int(candidate_coords[sorted_idx, 0].item())
            x = int(candidate_coords[sorted_idx, 1].item())
            score = float(candidate_scores[sorted_idx].item())

            is_too_close = False
            for selected_y, selected_x in selected_points:
                chebyshev_distance = max(abs(y - selected_y), abs(x - selected_x))
                if chebyshev_distance < int(min_distance):
                    is_too_close = True
                    break
            if is_too_close:
                continue

            selected_points.append((y, x))
            prompt_seed_list.append(
                {
                    "batch_idx": int(batch_idx),
                    "y": y,
                    "x": x,
                    "score": score,
                    "conflict_class": int(conflict_cls_map[batch_idx, y, x].item()),
                }
            )

            if len(selected_points) >= int(topk):
                break

    return prompt_seed_list


def generate_boundary_prompt_scores(
    feature_map: torch.Tensor,
    prob_map: torch.Tensor,
    pred_mask: torch.Tensor,
    prototype_library: Mapping[BoundaryKey, torch.Tensor],
    num_classes: int,
    organ_ids: Optional[Sequence[int]] = None,
    boundary_kernel: int = 3,
    topk: int = 10,
    min_score: float = 0.0,
    min_distance: int = 5,
    ordered: bool = False,
    missing_proto_value: float = 0.0,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, List[Dict[str, Union[int, float]]]], Dict[int, Dict[str, torch.Tensor]]]:
    """
    为所有前景器官生成 boundary prompt score maps，并选取 top-k 高分边界锚点。

    参数:
        feature_map: torch.Tensor
            特征图，形状为 `(B, C, H, W)`。
        prob_map: torch.Tensor
            粗分割概率图，形状为 `(B, K, H, W)`。
        pred_mask: torch.Tensor
            粗分割类别图，形状为 `(B, H, W)`。
        prototype_library: Mapping[BoundaryKey, torch.Tensor]
            细粒度边界单原型库。
        num_classes: int
            类别总数，包含背景 0。
        organ_ids: Optional[Sequence[int]]
            需要处理的器官类别列表。若为 None，则默认 `[1, ..., num_classes-1]`。
        boundary_kernel: int
            粗边界带核大小。
        topk: int
            每张图每个器官最多保留多少个点。
        min_score: float
            选点分数阈值。
        min_distance: int
            选点最小空间距离阈值。
        ordered: bool
            是否保留有序边界表示。
        missing_proto_value: float
            prototype 缺失时的相似度填充值。

    返回:
        score_maps_dict: Dict[int, torch.Tensor]
            每个器官对应一个 score map，value 形状为 `(B, H, W)`。
        prompt_seed_dict: Dict[int, List[Dict[str, Union[int, float]]]]
            每个器官对应一个高分边界锚点列表。
        aux_dict_per_organ: Dict[int, Dict[str, torch.Tensor]]
            每个器官对应一份中间结果字典。
    """
    feature_map = _ensure_feature_map_is_bchw(feature_map)
    prob_map = _ensure_prob_map_is_bkhw(prob_map)
    pred_mask = _ensure_label_is_bhw(pred_mask)

    if num_classes <= 1:
        raise ValueError("num_classes 必须大于 1")
    if prob_map.shape[1] != num_classes:
        raise ValueError(
            f"prob_map 的类别维度与 num_classes 不一致，prob_map.shape[1]={prob_map.shape[1]}, num_classes={num_classes}"
        )

    if organ_ids is None:
        organ_ids = list(range(1, num_classes))
    else:
        organ_ids = [int(organ_id) for organ_id in organ_ids if int(organ_id) > 0]

    # top1 / top2 与 uncertainty 对所有器官共享，先算一次，免得后面重复开销。
    top1_prob, top2_prob, top1_cls, top2_cls = extract_top1_top2(prob_map)

    score_maps_dict: Dict[int, torch.Tensor] = {}
    prompt_seed_dict: Dict[int, List[Dict[str, Union[int, float]]]] = {}
    aux_dict_per_organ: Dict[int, Dict[str, torch.Tensor]] = {}

    for organ_id in organ_ids:
        score_map, aux_dict = _compute_boundary_prompt_score_for_organ_from_top12(
            feature_map=feature_map,
            pred_mask=pred_mask,
            top1_prob=top1_prob,
            top2_prob=top2_prob,
            top1_cls=top1_cls,
            top2_cls=top2_cls,
            organ_id=organ_id,
            prototype_library=prototype_library,
            boundary_kernel=boundary_kernel,
            ordered=ordered,
            missing_proto_value=missing_proto_value,
        )
        prompt_seed_list = select_topk_prompt_seeds(
            score_map=score_map,
            conflict_cls_map=aux_dict["conflict_cls_map"],
            topk=topk,
            min_score=min_score,
            min_distance=min_distance,
        )

        score_maps_dict[int(organ_id)] = score_map
        prompt_seed_dict[int(organ_id)] = prompt_seed_list
        aux_dict_per_organ[int(organ_id)] = aux_dict

    return score_maps_dict, prompt_seed_dict, aux_dict_per_organ


def visualize_prompt_seeds_on_prediction(
    pred_mask_2d: torch.Tensor,
    prompt_seed_list: List[Dict[str, Union[int, float]]],
    organ_id: int,
    save_path: Union[str, Path],
) -> None:
    """
    将单张图上的高分边界锚点绘制在粗分割结果上。

    参数:
        pred_mask_2d: torch.Tensor
            单张粗分割类别图，形状为 `(H, W)`。
        prompt_seed_list: List[Dict[str, Union[int, float]]]
            当前器官的 prompt seed 列表。
        organ_id: int
            当前器官类别编号。
        save_path: Union[str, Path]
            可视化图像保存路径。
    """
    pred_mask_2d = _ensure_label_is_bhw(pred_mask_2d)[0].detach().cpu()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    organ_color = plt.get_cmap("tab10")(int(organ_id) % 10)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(pred_mask_2d.numpy(), cmap="tab20")
    ax.set_title(f"Prompt Seeds on Prediction (organ={organ_id})")
    ax.set_axis_off()

    # 这里只画当前器官的种子点，并顺手标出 conflict class，后面排查边界混淆时会很方便。
    filtered_seed_list = [seed for seed in prompt_seed_list if int(seed["batch_idx"]) == 0]
    for point_idx, seed_info in enumerate(filtered_seed_list):
        y = int(seed_info["y"])
        x = int(seed_info["x"])
        score = float(seed_info["score"])
        conflict_class = int(seed_info["conflict_class"])

        ax.scatter(
            x,
            y,
            s=64,
            c=[organ_color],
            edgecolors="white",
            linewidths=1.2,
            marker="o",
        )
        ax.text(
            x + 1.5,
            y - 1.5,
            f"{point_idx}:{conflict_class}\n{score:.3f}",
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5},
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _build_demo_pred_mask(
    height: int = 128,
    width: int = 128,
) -> torch.Tensor:
    """
    构造 demo 用的粗分割类别图。

    标签约定:
        - 0: 背景
        - 1: 左上器官
        - 2: 右上器官
        - 3: 下方器官

    参数:
        height: int
            图像高度。
        width: int
            图像宽度。

    返回:
        pred_mask: torch.Tensor
            形状为 `(1, H, W)` 的粗分割类别图。
    """
    pred_mask = torch.zeros((1, height, width), dtype=torch.long)
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )

    organ1_mask = (yy >= 18) & (yy <= 76) & (xx >= 18) & (xx <= 52)
    organ2_mask = (yy >= 24) & (yy <= 82) & (xx >= 53) & (xx <= 94)
    organ3_mask = ((yy - 94.0) / 18.0) ** 2 + ((xx - 50.0) / 28.0) ** 2 <= 1.0

    pred_mask[0][organ1_mask] = 1
    pred_mask[0][organ2_mask] = 2
    pred_mask[0][organ3_mask] = 3
    return pred_mask


def _build_demo_prob_map(
    pred_mask: torch.Tensor,
    num_classes: int,
    blur_kernel: int = 9,
) -> torch.Tensor:
    """
    基于粗分割类别图构造一个 demo 用的粗分割概率图。

    构造思路:
        1. 先将粗分割转为 one-hot
        2. 再在空间维做平均池化，模拟边界附近类别混合
        3. 最后重新归一化成概率分布

    参数:
        pred_mask: torch.Tensor
            粗分割类别图，形状为 `(B, H, W)`。
        num_classes: int
            类别总数，包含背景 0。
        blur_kernel: int
            平滑核大小。

    返回:
        prob_map: torch.Tensor
            构造好的概率图，形状为 `(B, K, H, W)`。
    """
    pred_mask = _ensure_label_is_bhw(pred_mask)
    blur_kernel = _validate_kernel_size(blur_kernel)

    one_hot = F.one_hot(pred_mask, num_classes=num_classes).permute(0, 3, 1, 2).to(dtype=torch.float32)
    prob_map = F.avg_pool2d(
        one_hot,
        kernel_size=blur_kernel,
        stride=1,
        padding=blur_kernel // 2,
    )
    prob_map = prob_map + 1e-6
    prob_map = prob_map / prob_map.sum(dim=1, keepdim=True)
    return prob_map


def _build_demo_feature_map(
    prob_map: torch.Tensor,
    feature_dim: int = 12,
) -> torch.Tensor:
    """
    基于概率图构造 demo 用的目标域特征图。

    构造思路:
        1. 用类别基向量按 `prob_map` 做加权和，模拟边界处的类别混合特征
        2. 叠加少量空间基特征，让图像内部也带一点位置结构
        3. 最后输出给 score 模块和 prototype 相似度模块复用

    参数:
        prob_map: torch.Tensor
            概率图，形状为 `(B, K, H, W)`。
        feature_dim: int
            特征通道数 C。

    返回:
        feature_map: torch.Tensor
            构造好的特征图，形状为 `(B, C, H, W)`。
    """
    prob_map = _ensure_prob_map_is_bkhw(prob_map)
    batch_size, num_classes, height, width = prob_map.shape

    class_feature_bank = torch.tensor(
        [
            [0.85, 0.10, 0.05, 0.00, 0.12, 0.03, 0.06, 0.02, 0.01, 0.04, 0.00, 0.05],
            [0.10, 0.90, 0.18, 0.05, 0.08, 0.15, 0.00, 0.12, 0.02, 0.06, 0.05, 0.00],
            [0.08, 0.22, 0.92, 0.10, 0.02, 0.14, 0.10, 0.03, 0.06, 0.00, 0.08, 0.05],
            [0.05, 0.08, 0.12, 0.94, 0.16, 0.00, 0.12, 0.04, 0.05, 0.10, 0.03, 0.06],
        ],
        dtype=torch.float32,
        device=prob_map.device,
    )
    if num_classes > class_feature_bank.shape[0]:
        raise ValueError(
            f"demo 当前只支持最多 {class_feature_bank.shape[0]} 个类别，当前 num_classes={num_classes}"
        )
    class_feature_bank = class_feature_bank[:num_classes, :feature_dim]

    # 按类别概率做加权和，让边界附近的特征天然更像“两个类别混合后的边界表征”。
    weighted_class_feature = torch.einsum("bkhw,kc->bchw", prob_map, class_feature_bank)

    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, steps=height, device=prob_map.device),
        torch.linspace(0.0, 1.0, steps=width, device=prob_map.device),
        indexing="ij",
    )
    spatial_basis = torch.stack(
        [
            yy,
            xx,
            torch.sin(math.pi * yy),
            torch.cos(math.pi * xx),
            torch.sin(math.pi * (yy + xx)),
            torch.cos(math.pi * (yy - xx)),
            yy * xx,
            yy ** 2,
            xx ** 2,
            torch.sin(2 * math.pi * yy),
            torch.cos(2 * math.pi * xx),
            torch.sin(2 * math.pi * (yy + xx)),
        ],
        dim=0,
    )[:feature_dim]
    spatial_feature = spatial_basis.unsqueeze(0).expand(batch_size, -1, -1, -1)

    feature_map = weighted_class_feature + 0.12 * spatial_feature
    return feature_map.to(dtype=torch.float32)


def _build_demo_prototype_library(
    num_classes: int,
    feature_dim: int,
    device: Union[str, torch.device] = "cpu",
) -> Dict[BoundaryKey, torch.Tensor]:
    """
    构造 demo 用的小型细粒度边界 prototype library。

    构造规则:
        对每个边界类别 `(A, B)`，prototype 近似设为两个类别基向量的平均后再归一化。

    参数:
        num_classes: int
            类别总数，包含背景 0。
        feature_dim: int
            prototype 维度 C。
        device: Union[str, torch.device]
            prototype 存放设备。

    返回:
        prototype_library: Dict[BoundaryKey, torch.Tensor]
            demo 用原型库。
    """
    device = torch.device(device)
    class_feature_bank = torch.tensor(
        [
            [0.85, 0.10, 0.05, 0.00, 0.12, 0.03, 0.06, 0.02, 0.01, 0.04, 0.00, 0.05],
            [0.10, 0.90, 0.18, 0.05, 0.08, 0.15, 0.00, 0.12, 0.02, 0.06, 0.05, 0.00],
            [0.08, 0.22, 0.92, 0.10, 0.02, 0.14, 0.10, 0.03, 0.06, 0.00, 0.08, 0.05],
            [0.05, 0.08, 0.12, 0.94, 0.16, 0.00, 0.12, 0.04, 0.05, 0.10, 0.03, 0.06],
        ],
        dtype=torch.float32,
        device=device,
    )[:num_classes, :feature_dim]

    prototype_library: Dict[BoundaryKey, torch.Tensor] = {}
    for class_a in range(1, num_classes):
        for class_b in range(num_classes):
            if class_a == class_b:
                continue
            pair_key = canonicalize_boundary_key(class_a, class_b, ordered=False)
            if pair_key in prototype_library:
                continue

            prototype = 0.5 * (class_feature_bank[class_a] + class_feature_bank[class_b])
            prototype = F.normalize(prototype.unsqueeze(0), p=2, dim=1, eps=1e-12).squeeze(0)
            prototype_library[pair_key] = prototype

    return prototype_library


def run_minimal_demo() -> Dict[str, Any]:
    """
    运行最小可执行 demo，验证目标域 boundary prompt score 生成流程是否跑通。

    demo 步骤:
        1. 构造 dummy pred_mask
        2. 构造 dummy prob_map
        3. 构造 dummy feature_map
        4. 构造小型 prototype_library
        5. 运行 `generate_boundary_prompt_scores`
        6. 打印各器官高分点数量
        7. 保存一张高分边界锚点可视化图

    返回:
        demo_result: Dict[str, Any]
            包含 score_maps_dict、prompt_seed_dict、aux_dict_per_organ 和可视化路径。
    """
    torch.manual_seed(2026)

    num_classes = 4
    feature_dim = 12
    pred_mask = _build_demo_pred_mask(height=128, width=128)
    prob_map = _build_demo_prob_map(pred_mask=pred_mask, num_classes=num_classes, blur_kernel=9)
    pred_mask = prob_map.argmax(dim=1)
    feature_map = _build_demo_feature_map(prob_map=prob_map, feature_dim=feature_dim)
    prototype_library = _build_demo_prototype_library(
        num_classes=num_classes,
        feature_dim=feature_dim,
        device=feature_map.device,
    )

    score_maps_dict, prompt_seed_dict, aux_dict_per_organ = generate_boundary_prompt_scores(
        feature_map=feature_map,
        prob_map=prob_map,
        pred_mask=pred_mask,
        prototype_library=prototype_library,
        num_classes=num_classes,
        organ_ids=None,
        boundary_kernel=3,
        topk=8,
        min_score=0.05,
        min_distance=6,
        ordered=False,
        missing_proto_value=0.0,
    )

    print("=" * 100)
    print("目标域 boundary prompt score demo")
    print("=" * 100)
    for organ_id in sorted(prompt_seed_dict.keys()):
        print(f"organ_id={organ_id}, num_selected_seeds={len(prompt_seed_dict[organ_id])}")

    save_dir = Path(__file__).resolve().parent
    visualization_path = save_dir / "demo_target_boundary_prompt_scores_organ1.png"
    visualize_prompt_seeds_on_prediction(
        pred_mask_2d=pred_mask[0],
        prompt_seed_list=prompt_seed_dict.get(1, []),
        organ_id=1,
        save_path=visualization_path,
    )
    print(f"visualization saved to: {visualization_path}")

    return {
        "score_maps_dict": score_maps_dict,
        "prompt_seed_dict": prompt_seed_dict,
        "aux_dict_per_organ": aux_dict_per_organ,
        "visualization_path": str(visualization_path),
    }


if __name__ == "__main__":
    run_minimal_demo()
