"""目标域有序细粒度边界 score 模块。

本文件只处理“第三部分”里的边界响应计算：
1. 从有序边界 prototype bank 中取出 prototype。
2. 用目标域特征图与有序 prototype 逐类计算余弦相似度响应图。
3. 显式区分 (A, B) 与 (B, A) 两个方向的边界 score。

注意：
1. 当前文件只做 ordered boundary score，不做 SAM prompt、SAM refinement 或任何目标域训练损失。
2. 当前文件默认输入的是某层目标域 feature_map，以及已经建好的 ordered prototype bank。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from dynamic_boundary_prototype_bank import DynamicBoundaryPrototypeBank


BoundaryKey = Tuple[int, int]
BoundaryScoreDict = Dict[BoundaryKey, torch.Tensor]
BoundaryMaskDict = Dict[BoundaryKey, torch.Tensor]


def canonicalize_ordered_boundary_key(a: int, b: int) -> BoundaryKey:
    """规范化有序细粒度边界 key。

    输入：
        a: int
            当前边界点所属类别编号。
        b: int
            当前边界点局部邻域接触到的类别编号。

    输出：
        boundary_key: Tuple[int, int]
            有序边界 key，固定返回 (a, b)。

    说明：
        1. 当前模块只支持 ordered boundary。
        2. (A, B) 与 (B, A) 一定会被当成两个不同的响应目标。
    """
    return int(a), int(b)


def _l2_normalize_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    """对特征图按通道做 L2 normalize。

    输入：
        feature_map: Tensor, shape = (B, C, H, W)
            目标域特征图。

    输出：
        normalized_feature_map: Tensor, shape = (B, C, H, W)
            每个空间位置的通道向量都做过 L2 normalize 的特征图。
    """
    if feature_map.dim() != 4:
        raise ValueError("feature_map must have shape (B, C, H, W)")
    return F.normalize(feature_map.to(torch.float32), p=2, dim=1, eps=1e-12)


def _l2_normalize_prototype(prototype: torch.Tensor) -> torch.Tensor:
    """对单个 prototype 向量做 L2 normalize。"""
    if prototype.dim() != 1:
        raise ValueError("prototype must have shape (C,)")
    return F.normalize(prototype.unsqueeze(0), p=2, dim=1, eps=1e-12).squeeze(0)


def extract_ordered_prototype_dict(
    prototype_source: DynamicBoundaryPrototypeBank | Dict[BoundaryKey, Dict[str, torch.Tensor | int]] | Dict[BoundaryKey, torch.Tensor],
) -> Dict[BoundaryKey, torch.Tensor]:
    """把不同形式的 prototype 容器统一整理成 ordered prototype 字典。

    输入：
        prototype_source:
            支持以下两种输入：
            1. DynamicBoundaryPrototypeBank
            2. dict
               - {(A, B): {"prototype": Tensor[C], ...}}
               - {(A, B): Tensor[C]}

    输出：
        ordered_prototype_dict: dict
            形式为：
            {
                (A, B): Tensor[C],
                ...
            }
    """
    ordered_prototype_dict: Dict[BoundaryKey, torch.Tensor] = {}

    if isinstance(prototype_source, DynamicBoundaryPrototypeBank):
        for raw_key, entry in prototype_source.bank.items():
            boundary_key = canonicalize_ordered_boundary_key(raw_key[0], raw_key[1])
            prototype = entry["prototype"]
            if not isinstance(prototype, torch.Tensor):
                raise TypeError(f"bank prototype for boundary {boundary_key} must be a torch.Tensor")
            ordered_prototype_dict[boundary_key] = _l2_normalize_prototype(prototype.detach().to(torch.float32))
        return ordered_prototype_dict

    if not isinstance(prototype_source, dict):
        raise TypeError("prototype_source must be a DynamicBoundaryPrototypeBank or a dictionary")

    for raw_key, entry in prototype_source.items():
        boundary_key = canonicalize_ordered_boundary_key(raw_key[0], raw_key[1])
        if isinstance(entry, torch.Tensor):
            prototype = entry
        elif isinstance(entry, dict):
            prototype = entry.get("prototype")
            if not isinstance(prototype, torch.Tensor):
                raise TypeError(f"prototype_source[{boundary_key}]['prototype'] must be a torch.Tensor")
        else:
            raise TypeError(f"prototype_source[{boundary_key}] must be a Tensor or a dictionary")
        ordered_prototype_dict[boundary_key] = _l2_normalize_prototype(prototype.detach().to(torch.float32))

    return ordered_prototype_dict


def compute_ordered_boundary_score_map(
    feature_map: torch.Tensor,
    prototype: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
    fill_value: float = -1.0,
) -> torch.Tensor:
    """计算单个有序边界类别的余弦相似度响应图。

    输入：
        feature_map: Tensor, shape = (B, C, H, W)
            目标域特征图。
        prototype: Tensor, shape = (C,)
            某个有序边界类别 (A, B) 的 prototype。
        candidate_mask: Optional[Tensor], shape = (B, H, W)
            当前有序边界对应的一侧候选区域。
            - 若提供，则只在 mask=True 的位置保留 score。
            - 若不提供，则对整张图都计算响应。
        fill_value: float
            在候选区域外部填充的数值。

    输出：
        score_map: Tensor, shape = (B, H, W)
            当前 ordered boundary 的余弦相似度响应图。
    """
    normalized_feature_map = _l2_normalize_feature_map(feature_map)
    normalized_prototype = _l2_normalize_prototype(prototype.to(feature_map.device, dtype=torch.float32))
    score_map = torch.einsum("bchw,c->bhw", normalized_feature_map, normalized_prototype)
    if candidate_mask is not None:
        if candidate_mask.shape != score_map.shape:
            raise ValueError("candidate_mask must have shape (B, H, W) and match the score map shape")
        candidate_mask = candidate_mask.to(device=score_map.device, dtype=torch.bool)
        score_map = torch.where(candidate_mask, score_map, torch.full_like(score_map, float(fill_value)))
    return score_map


def compute_all_ordered_boundary_scores(
    feature_map: torch.Tensor,
    prototype_source: DynamicBoundaryPrototypeBank | Dict[BoundaryKey, Dict[str, torch.Tensor | int]] | Dict[BoundaryKey, torch.Tensor],
    target_keys: Optional[Iterable[BoundaryKey]] = None,
    candidate_mask_dict: Optional[BoundaryMaskDict] = None,
    fill_value: float = -1.0,
) -> BoundaryScoreDict:
    """为全部或指定的有序边界类别计算响应图。

    输入：
        feature_map: Tensor, shape = (B, C, H, W)
            目标域特征图。
        prototype_source:
            有序 prototype bank 或 prototype 字典。
        target_keys: Optional[Iterable[BoundaryKey]]
            若为 None，则对 prototype_source 中的全部 ordered key 计算响应图。
            若不为 None，则只计算指定 ordered key 的响应图。
        candidate_mask_dict: Optional[dict]
            每个有序边界 key 对应的一侧候选区域 mask。
            例如：
            {
                (A, B): Tensor[B, H, W],   # A 侧候选区域
                (B, A): Tensor[B, H, W],   # B 侧候选区域
            }
        fill_value: float
            候选区域外部填充值。

    输出：
        score_dict: dict
            形式为：
            {
                (A, B): Tensor[B, H, W],
                ...
            }

    说明：
        1. 当前函数不会把 (A, B) 与 (B, A) 合并。
        2. 如果同时给出 (A, B) 与 (B, A)，会分别输出两张 score map。
    """
    ordered_prototype_dict = extract_ordered_prototype_dict(prototype_source)
    if target_keys is None:
        used_keys = sorted(ordered_prototype_dict.keys())
    else:
        used_keys = [canonicalize_ordered_boundary_key(key[0], key[1]) for key in target_keys]

    score_dict: BoundaryScoreDict = {}
    for boundary_key in used_keys:
        if boundary_key not in ordered_prototype_dict:
            continue
        score_dict[boundary_key] = compute_ordered_boundary_score_map(
            feature_map=feature_map,
            prototype=ordered_prototype_dict[boundary_key],
            candidate_mask=None if candidate_mask_dict is None else candidate_mask_dict.get(boundary_key),
            fill_value=fill_value,
        )
    return score_dict


def compute_bidirectional_boundary_pair_scores(
    feature_map: torch.Tensor,
    prototype_source: DynamicBoundaryPrototypeBank | Dict[BoundaryKey, Dict[str, torch.Tensor | int]] | Dict[BoundaryKey, torch.Tensor],
    class_a: int,
    class_b: int,
    candidate_mask_dict: Optional[BoundaryMaskDict] = None,
    fill_value: float = -1.0,
) -> BoundaryScoreDict:
    """分别计算 (A, B) 与 (B, A) 两个方向的边界响应图。

    输入：
        feature_map: Tensor, shape = (B, C, H, W)
            目标域特征图。
        prototype_source:
            有序 prototype bank 或 prototype 字典。
        class_a: int
            第一个类别编号。
        class_b: int
            第二个类别编号。
        candidate_mask_dict: Optional[dict]
            双向 ordered boundary 的候选区域 mask 字典。
            - `(A, B)` 对应 A 侧候选区域
            - `(B, A)` 对应 B 侧候选区域
        fill_value: float
            候选区域外部填充值。

    输出：
        score_dict: dict
            形式为：
            {
                (A, B): Tensor[B, H, W],   # 若 prototype 存在
                (B, A): Tensor[B, H, W],   # 若 prototype 存在
            }

    说明：
        1. 这是当前模块最直接体现“有序定义”的函数。
        2. 它不会返回一个合并后的 A-B score，只会分别返回：
           - score_(A,B)
           - score_(B,A)
    """
    forward_key = canonicalize_ordered_boundary_key(class_a, class_b)
    backward_key = canonicalize_ordered_boundary_key(class_b, class_a)
    return compute_all_ordered_boundary_scores(
        feature_map=feature_map,
        prototype_source=prototype_source,
        target_keys=(forward_key, backward_key),
        candidate_mask_dict=candidate_mask_dict,
        fill_value=fill_value,
    )


def summarize_ordered_boundary_scores(score_dict: BoundaryScoreDict) -> Dict[BoundaryKey, Dict[str, float | Tuple[int, ...]]]:
    """汇总 ordered boundary score 的基础统计信息。

    输入：
        score_dict: dict
            compute_all_ordered_boundary_scores 或 compute_bidirectional_boundary_pair_scores 的输出。

    输出：
        summary_dict: dict
            形式为：
            {
                (A, B): {
                    "shape": (B, H, W),
                    "min": float,
                    "max": float,
                    "mean": float,
                }
            }
    """
    summary_dict: Dict[BoundaryKey, Dict[str, float | Tuple[int, ...]]] = {}
    for boundary_key in sorted(score_dict.keys()):
        score_map = score_dict[boundary_key]
        if not isinstance(score_map, torch.Tensor):
            raise TypeError(f"score_dict[{boundary_key}] must be a torch.Tensor")
        summary_dict[boundary_key] = {
            "shape": tuple(score_map.shape),
            "min": float(score_map.min().item()),
            "max": float(score_map.max().item()),
            "mean": float(score_map.mean().item()),
        }
    return summary_dict


def extract_ordered_boundary_prompt_seeds(
    score_dict: BoundaryScoreDict,
    topk_per_boundary: int = 10,
    min_score: Optional[float] = None,
) -> Dict[BoundaryKey, List[Dict[str, int | float | BoundaryKey]]]:
    """从有序边界 score 图中提取 ordered seed。

    输入：
        score_dict: dict
            形式为：
            {
                (A, B): Tensor[B, H, W],
                ...
            }
        topk_per_boundary: int
            每个有序边界类别最多保留多少个高分 seed。
        min_score: Optional[float]
            最低分数阈值。
            - 若为 None，则只按 top-k 截断。
            - 若不为 None，则先过滤低于阈值的位置。

    输出：
        seed_dict: dict
            形式为：
            {
                (A, B): [
                    {
                        "ordered_boundary_key": (A, B),
                        "seed_side_class": A,
                        "neighbor_class": B,
                        "batch_idx": int,
                        "y": int,
                        "x": int,
                        "score": float,
                    },
                    ...
                ]
            }

    说明：
        1. 当前 seed 结构显式保留 ordered_boundary_key，不再只给一个无方向 conflict_class。
        2. 对于 (A, B)，输出 seed 明确表示“这个 seed 属于 A 侧，面向 B”。
    """
    if topk_per_boundary < 1:
        raise ValueError("topk_per_boundary must be >= 1")

    seed_dict: Dict[BoundaryKey, List[Dict[str, int | float | BoundaryKey]]] = {}
    for boundary_key in sorted(score_dict.keys()):
        score_map = score_dict[boundary_key]
        if not isinstance(score_map, torch.Tensor):
            raise TypeError(f"score_dict[{boundary_key}] must be a torch.Tensor")
        if score_map.dim() != 3:
            raise ValueError(f"score_dict[{boundary_key}] must have shape (B, H, W)")

        flat_score = score_map.reshape(-1)
        if min_score is not None:
            valid_mask = flat_score >= float(min_score)
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            if valid_indices.numel() == 0:
                seed_dict[boundary_key] = []
                continue
            valid_scores = flat_score[valid_indices]
            keep_count = min(int(topk_per_boundary), int(valid_indices.numel()))
            top_indices_in_valid = torch.topk(valid_scores, k=keep_count, largest=True).indices
            top_flat_indices = valid_indices[top_indices_in_valid]
        else:
            keep_count = min(int(topk_per_boundary), int(flat_score.numel()))
            top_flat_indices = torch.topk(flat_score, k=keep_count, largest=True).indices

        batch_size, height, width = score_map.shape
        seeds_this_key: List[Dict[str, int | float | BoundaryKey]] = []
        for flat_index in top_flat_indices.tolist():
            batch_idx = int(flat_index // (height * width))
            remain = int(flat_index % (height * width))
            y = int(remain // width)
            x = int(remain % width)
            seeds_this_key.append(
                {
                    "ordered_boundary_key": boundary_key,
                    "seed_side_class": int(boundary_key[0]),
                    "neighbor_class": int(boundary_key[1]),
                    "batch_idx": batch_idx,
                    "y": y,
                    "x": x,
                    "score": float(score_map[batch_idx, y, x].item()),
                }
            )
        seed_dict[boundary_key] = seeds_this_key

    return seed_dict
