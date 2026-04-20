import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


TensorLike = Union[torch.Tensor, np.ndarray]
BoundaryKey = Tuple[int, int]


def _validate_kernel_size(kernel_size: int) -> int:
    """
    校验形态学或邻域统计所使用的核大小是否合法。

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
    将输入二值 mask 统一整理成 `(B, 1, H, W)` 形式，便于后续使用 PyTorch 池化实现形态学操作。

    参数:
        mask: torch.Tensor
            支持三种输入形状：
            1. `(H, W)`
            2. `(N, H, W)`，这里的 `N` 可以理解为批大小
            3. `(N, 1, H, W)`

    返回:
        mask_4d: torch.Tensor
            统一后的 4 维张量，形状为 `(B, 1, H, W)`，数值为 0/1 float。
        original_ndim: int
            原始输入维度，用于在函数末尾恢复形状。
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
        raise ValueError(f"mask 维度必须是 2/3/4，当前 ndim={mask.ndim}")

    # 这里统一转成 float，后面 max_pool2d / min-pool 替代实现会更顺手。
    mask_4d = (mask_4d > 0).to(dtype=torch.float32)
    return mask_4d, original_ndim


def _restore_binary_mask(mask_4d: torch.Tensor, original_ndim: int) -> torch.Tensor:
    """
    将 `(B, 1, H, W)` 形式的二值张量恢复回调用者最初的输入维度。

    参数:
        mask_4d: torch.Tensor
            形状为 `(B, 1, H, W)` 的布尔或 0/1 张量。
        original_ndim: int
            调用前输入张量的原始维度。

    返回:
        restored_mask: torch.Tensor
            恢复维度后的张量：
            1. 原始是 `(H, W)`，则返回 `(H, W)`
            2. 原始是 `(N, H, W)`，则返回 `(N, H, W)`
            3. 原始是 `(N, 1, H, W)`，则返回 `(N, 1, H, W)`
    """
    if original_ndim == 2:
        return mask_4d[0, 0]
    if original_ndim == 3:
        return mask_4d[:, 0]
    if original_ndim == 4:
        return mask_4d
    raise ValueError(f"不支持的 original_ndim={original_ndim}")


def _ensure_label_is_bhw(seg_label: torch.Tensor) -> torch.Tensor:
    """
    将标签张量整理为 `(B, H, W)` 形式。

    参数:
        seg_label: torch.Tensor
            支持 `(H, W)` 或 `(B, H, W)`。

    返回:
        seg_label_bhw: torch.Tensor
            形状统一为 `(B, H, W)` 的 long 张量。
    """
    if not isinstance(seg_label, torch.Tensor):
        raise TypeError("seg_label 必须是 torch.Tensor")

    if seg_label.ndim == 2:
        seg_label = seg_label.unsqueeze(0)
    elif seg_label.ndim != 3:
        raise ValueError(f"seg_label 必须是 2/3 维张量，当前 shape={tuple(seg_label.shape)}")

    return seg_label.long()


def canonicalize_boundary_pair(
    class_a: int,
    class_b: int,
    ignore_background_order: bool = True,
) -> BoundaryKey:
    """
    将边界类别统一规范成固定的 tuple key，避免同一条边界被 `(A, B)` / `(B, A)` 重复表示。

    设计决定:
        1. 器官-器官边界采用无序表示，即 `(min(A, B), max(A, B))`
        2. 器官-背景边界统一写成 `(A, 0)`，保持前景类别在前、背景 0 在后

    参数:
        class_a: int
            当前点所属类别。
        class_b: int
            邻域中选中的相邻类别。
        ignore_background_order: bool
            是否对背景边界采取固定顺序表示。这里默认 True。

    返回:
        pair_key: Tuple[int, int]
            规范化后的边界类别 key。
    """
    if class_a == class_b:
        raise ValueError("边界对中的两个类别不能相同")

    if ignore_background_order and (class_a == 0 or class_b == 0):
        foreground_class = class_b if class_a == 0 else class_a
        return int(foreground_class), 0

    return tuple(sorted((int(class_a), int(class_b))))


def extract_morph_boundary(
    mask: torch.Tensor,
    kernel_size: int = 3,
    mode: str = "inner",
) -> torch.Tensor:
    """
    使用 PyTorch 形态学近似操作快速提取单类二值 mask 的边界带。

    支持两种模式:
        1. `inner`: `mask - erode(mask)`，返回 mask 内侧的一圈边界
        2. `gradient`: `dilate(mask) - erode(mask)`，返回更宽一些的形态学梯度边界

    参数:
        mask: torch.Tensor
            单类二值 mask，支持以下输入形状：
            1. `(H, W)`
            2. `(N, H, W)`
            3. `(N, 1, H, W)`
        kernel_size: int
            形态学核大小，必须为正奇数，默认 3。
        mode: str
            边界提取模式，支持 `"inner"` 或 `"gradient"`。

    返回:
        boundary_mask: torch.Tensor
            与输入维度对齐的边界带 mask，数值类型为 bool。
    """
    kernel_size = _validate_kernel_size(kernel_size)
    mask_4d, original_ndim = _prepare_binary_mask(mask)
    padding = kernel_size // 2

    # 膨胀可以直接用 max_pool2d 来近似实现，速度快，也方便放到 GPU 上。
    dilated = F.max_pool2d(mask_4d, kernel_size=kernel_size, stride=1, padding=padding)

    # 腐蚀等价于 min-pool，这里借助 `-max_pool2d(-x)` 来实现。
    eroded = -F.max_pool2d(-mask_4d, kernel_size=kernel_size, stride=1, padding=padding)

    if mode == "inner":
        # inner boundary 只保留目标区域内部贴着边缘的一圈点。
        boundary = (mask_4d - eroded) > 0
    elif mode == "gradient":
        # gradient 边界更宽，会同时覆盖目标外侧和内侧一圈。
        boundary = (dilated - eroded) > 0
    else:
        raise ValueError(f"mode 只支持 'inner' 或 'gradient'，当前为 {mode}")

    return _restore_binary_mask(boundary, original_ndim)


def _build_neighborhood_class_count_map(
    seg_label: torch.Tensor,
    num_classes: int,
    kernel_size: int,
) -> torch.Tensor:
    """
    为每个像素预先统计其局部邻域内各类别像素的出现次数。

    参数:
        seg_label: torch.Tensor
            形状为 `(B, H, W)` 的标签图。
        num_classes: int
            总类别数，包含背景 0。
        kernel_size: int
            邻域统计核大小，必须为正奇数。

    返回:
        class_count_map: torch.Tensor
            形状为 `(B, num_classes, H, W)`。
            `class_count_map[b, c, y, x]` 表示像素 `(b, y, x)` 的 `kernel_size x kernel_size`
            邻域内，类别 `c` 出现了多少次。
    """
    kernel_size = _validate_kernel_size(kernel_size)
    seg_label = _ensure_label_is_bhw(seg_label)

    # one-hot 后形状变成 `(B, H, W, K)`，再转成卷积更顺手的 `(B, K, H, W)`。
    one_hot = F.one_hot(seg_label, num_classes=num_classes).permute(0, 3, 1, 2).to(torch.float32)

    # 这里用 group convolution 一次性统计所有类别的邻域计数，省掉像素级循环。
    weight = torch.ones(
        size=(num_classes, 1, kernel_size, kernel_size),
        device=seg_label.device,
        dtype=torch.float32,
    )
    class_count_map = F.conv2d(
        one_hot,
        weight=weight,
        bias=None,
        stride=1,
        padding=kernel_size // 2,
        groups=num_classes,
    )
    return class_count_map


def _append_coords_to_boundary_bucket(
    boundary_bucket: Dict[BoundaryKey, List[torch.Tensor]],
    pair_key: BoundaryKey,
    coords: torch.Tensor,
) -> None:
    """
    将某一类边界点坐标追加到指定边界类别的缓存列表中。

    参数:
        boundary_bucket: dict
            临时缓存字典，value 是若干个坐标张量列表。
        pair_key: Tuple[int, int]
            细粒度边界类别，如 `(1, 2)` 或 `(3, 0)`。
        coords: torch.Tensor
            当前要追加的坐标，形状为 `(N, 3)`，列顺序为 `(batch_idx, y, x)`。
    """
    if coords.numel() == 0:
        return
    boundary_bucket.setdefault(pair_key, []).append(coords)


def assign_fine_boundary_labels(
    seg_label: torch.Tensor,
    num_classes: int,
    kernel_size: int = 3,
    ignore_background_order: bool = True,
) -> Dict[BoundaryKey, Dict[str, torch.Tensor]]:
    """
    基于源域真实标签，为候选边界带中的点分配细粒度边界类别。

    整体流程:
        1. 对每个前景类别 A 提取其 inner boundary 候选带
        2. 仅在该候选带中检查局部邻域内出现了哪些类别
        3. 若存在其他前景类别，则优先归到器官-器官边界 `(A, B)`
        4. 若不存在其他前景类别，但邻域中存在背景，则归到器官-背景边界 `(A, 0)`

    归类策略设计决定:
        1. 一个点只属于一个边界类别，不允许多标签
        2. 当邻域里同时出现多个前景类别时，选择“邻域接壤像素数量最多”的类别
        3. 若多个前景类别接壤数量并列，使用 `torch.argmax` 的默认行为，保留类别编号更小者
        4. 器官-器官边界统一采用无序 pair key，例如 `(1, 2)` 和 `(2, 1)` 最终都会归到 `(1, 2)`
        5. 器官-背景边界统一写成 `(A, 0)`

    参数:
        seg_label: torch.Tensor
            真实标签图，形状为 `(B, H, W)` 或 `(H, W)`。
        num_classes: int
            总类别数，包含背景 0。
        kernel_size: int
            候选边界提取与邻域统计所使用的局部核大小。
        ignore_background_order: bool
            是否将背景边界固定写成 `(A, 0)`。

    返回:
        boundary_dict: Dict[Tuple[int, int], Dict[str, torch.Tensor]]
            每个 key 对应一个细粒度边界类别，value 中包含：
            - `coords`: `(N, 3)`，列顺序为 `(batch_idx, y, x)`
            - `batch_idx`: `(N,)`
            - `y`: `(N,)`
            - `x`: `(N,)`
            - `raw_count`: int，几何候选点数量
            - `pair_key`: `(A, B)` 形式的 tuple
    """
    kernel_size = _validate_kernel_size(kernel_size)
    seg_label = _ensure_label_is_bhw(seg_label)
    device = seg_label.device

    if num_classes <= 1:
        raise ValueError("num_classes 必须大于 1，至少应包含背景和一个前景类别")

    # 先一次性统计出所有像素局部邻域内的类别出现次数，后面所有类别都复用这张图。
    class_count_map = _build_neighborhood_class_count_map(seg_label, num_classes, kernel_size)
    boundary_bucket: Dict[BoundaryKey, List[torch.Tensor]] = {}

    for class_a in range(1, num_classes):
        # 每次只关注一个前景类别 A，先把属于 A 的区域抠出来。
        class_a_mask = seg_label.eq(class_a)
        if not bool(class_a_mask.any()):
            continue

        # 几何候选边界只在 A 类区域内部取一圈内边界，避免在整张图上做无意义扫描。
        candidate_boundary = extract_morph_boundary(
            class_a_mask,
            kernel_size=kernel_size,
            mode="inner",
        ).bool()
        if not bool(candidate_boundary.any()):
            continue

        # 这里取出所有前景类别的邻域计数，再把自身类别 A 的计数置零。
        # 这样 `max` 出来的就是“邻域中与 A 接壤最明显的其他前景类”。
        foreground_counts = class_count_map[:, 1:, :, :].clone()
        foreground_counts[:, class_a - 1, :, :] = 0
        max_foreground_count, best_foreground_index = foreground_counts.max(dim=1)
        best_foreground_class = best_foreground_index + 1

        # 背景邻接单独拿出来，只有“没有其他前景接壤”时才退化为 `(A, 0)`。
        background_count = class_count_map[:, 0, :, :]

        organ_boundary_mask = candidate_boundary & (max_foreground_count > 0)
        background_boundary_mask = candidate_boundary & (~organ_boundary_mask) & (background_count > 0)

        # 这里的循环只落在“类别数”层面，不是像素级循环，代价很低。
        for class_b in range(1, num_classes):
            if class_b == class_a:
                continue

            pair_mask = organ_boundary_mask & (best_foreground_class == class_b)
            if not bool(pair_mask.any()):
                continue

            coords = torch.nonzero(pair_mask, as_tuple=False)
            pair_key = canonicalize_boundary_pair(
                class_a=class_a,
                class_b=class_b,
                ignore_background_order=ignore_background_order,
            )
            _append_coords_to_boundary_bucket(boundary_bucket, pair_key, coords)

        if bool(background_boundary_mask.any()):
            coords = torch.nonzero(background_boundary_mask, as_tuple=False)
            pair_key = canonicalize_boundary_pair(
                class_a=class_a,
                class_b=0,
                ignore_background_order=ignore_background_order,
            )
            _append_coords_to_boundary_bucket(boundary_bucket, pair_key, coords)

    boundary_dict: Dict[BoundaryKey, Dict[str, torch.Tensor]] = {}
    for pair_key, coord_chunks in boundary_bucket.items():
        coords = torch.cat(coord_chunks, dim=0)
        boundary_dict[pair_key] = {
            "coords": coords,
            "batch_idx": coords[:, 0],
            "y": coords[:, 1],
            "x": coords[:, 2],
            "raw_count": int(coords.shape[0]),
            "pair_key": pair_key,
        }

    return boundary_dict


def gather_point_features(feature_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    按照给定坐标，从特征图中批量提取对应位置的特征向量。

    参数:
        feature_map: torch.Tensor
            特征图，形状为 `(B, C, H, W)`。
        coords: torch.Tensor
            坐标张量，形状为 `(N, 3)`，列顺序为 `(batch_idx, y, x)`。

    返回:
        point_features: torch.Tensor
            取出的点特征，形状为 `(N, C)`。
    """
    if feature_map.ndim != 4:
        raise ValueError(f"feature_map 必须是 4 维张量，当前 shape={tuple(feature_map.shape)}")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords 必须是 `(N, 3)` 形状，当前 shape={tuple(coords.shape)}")

    if coords.numel() == 0:
        return feature_map.new_zeros((0, feature_map.shape[1]))

    coords = coords.long()
    batch_idx = coords[:, 0]
    y = coords[:, 1]
    x = coords[:, 2]

    # 高级索引可以一次性把所有点特征取出来，没必要老老实实地一行行循环。
    point_features = feature_map[batch_idx, :, y, x]
    return point_features


def filter_boundary_points_by_feature_consistency(
    boundary_dict: Dict[BoundaryKey, Dict[str, torch.Tensor]],
    feature_map: torch.Tensor,
    keep_ratio: float = 0.8,
    min_points: int = 10,
) -> Dict[BoundaryKey, Dict[str, Union[int, float, bool, torch.Tensor, Tuple[int, int], Dict[str, float]]]]:
    """
    基于特征一致性，对每类几何候选边界点做轻量净化筛选。

    轻量版设计:
        1. 先提取每个边界类别 `(A, B)` 的所有候选点特征
        2. 对点特征做 L2 normalize
        3. 使用该类别候选点特征均值作为临时 prototype
        4. 计算每个点与 prototype 的余弦相似度
        5. 仅保留相似度最高的前 `keep_ratio` 比例点

    设计决定:
        当某一类边界点太少，小于 `min_points` 时，不做筛选，直接全部保留。
        理由很简单：样本太少时，临时 prototype 的统计稳定性本来就差，硬筛通常只会更吵。

    参数:
        boundary_dict: dict
            `assign_fine_boundary_labels` 的输出。
        feature_map: torch.Tensor
            特征图，形状为 `(B, C, H, W)`。
        keep_ratio: float
            保留比例，范围 `(0, 1]`。
        min_points: int
            低于该阈值时跳过筛选，全部保留。

    返回:
        filtered_boundary_dict: dict
            与输入相同的边界类别字典，但每一类都额外包含：
            - `features`: `(N_keep, C)`，筛选后保留的原始特征
            - `features_norm`: `(N_keep, C)`，L2 归一化后的保留特征
            - `prototype`: `(C,)`，该边界类别的临时原型
            - `cosine_similarity_all`: `(N_raw,)`
            - `cosine_similarity_kept`: `(N_keep,)`
            - `raw_count`: int
            - `kept_count`: int
            - `skipped_filtering`: bool
            - `similarity_stats`: dict，包含 mean/std/min/max/threshold
    """
    if feature_map.ndim != 4:
        raise ValueError(f"feature_map 必须是 4 维张量，当前 shape={tuple(feature_map.shape)}")
    if not (0 < keep_ratio <= 1):
        raise ValueError(f"keep_ratio 必须在 (0, 1] 范围内，当前为 {keep_ratio}")
    if min_points < 1:
        raise ValueError(f"min_points 必须 >= 1，当前为 {min_points}")

    filtered_boundary_dict = {}

    for pair_key, boundary_info in boundary_dict.items():
        coords = boundary_info["coords"].long()
        raw_count = int(coords.shape[0])
        if raw_count == 0:
            continue

        # 先从特征图里把该边界类别对应的候选点特征批量取出来。
        raw_features = gather_point_features(feature_map, coords).to(torch.float32)

        # L2 normalize 后，点积就等于余弦相似度，算起来干净又直接。
        normalized_features = F.normalize(raw_features, p=2, dim=1, eps=1e-12)

        # 临时 prototype 用所有候选点的均值来近似，然后再归一化。
        prototype = normalized_features.mean(dim=0, keepdim=False)
        prototype = F.normalize(prototype.unsqueeze(0), p=2, dim=1, eps=1e-12).squeeze(0)

        # 由于特征和原型都已归一化，这里直接矩阵乘就行，不需要额外 cosine API。
        cosine_similarity_all = torch.matmul(normalized_features, prototype)

        if raw_count < min_points:
            # 点太少时不筛选，全部留下，并显式记录“跳过筛选”。
            keep_indices = torch.arange(raw_count, device=coords.device)
            kept_count = raw_count
            skipped_filtering = True
            similarity_threshold = float(cosine_similarity_all.min().item())
        else:
            kept_count = max(1, int(math.ceil(raw_count * keep_ratio)))
            cosine_similarity_kept, keep_indices = torch.topk(
                cosine_similarity_all,
                k=kept_count,
                largest=True,
                sorted=True,
            )
            skipped_filtering = False
            similarity_threshold = float(cosine_similarity_kept[-1].item())

        kept_coords = coords.index_select(0, keep_indices)
        kept_features = raw_features.index_select(0, keep_indices)
        kept_features_norm = normalized_features.index_select(0, keep_indices)
        cosine_similarity_kept = cosine_similarity_all.index_select(0, keep_indices)

        filtered_boundary_dict[pair_key] = {
            "pair_key": pair_key,
            "coords": kept_coords,
            "batch_idx": kept_coords[:, 0],
            "y": kept_coords[:, 1],
            "x": kept_coords[:, 2],
            "features": kept_features,
            "features_norm": kept_features_norm,
            "prototype": prototype,
            "cosine_similarity_all": cosine_similarity_all,
            "cosine_similarity_kept": cosine_similarity_kept,
            "raw_count": raw_count,
            "kept_count": int(kept_coords.shape[0]),
            "skipped_filtering": skipped_filtering,
            "keep_ratio": keep_ratio,
            "min_points": min_points,
            "similarity_stats": {
                "mean": float(cosine_similarity_all.mean().item()),
                "std": float(cosine_similarity_all.std(unbiased=False).item()),
                "min": float(cosine_similarity_all.min().item()),
                "max": float(cosine_similarity_all.max().item()),
                "threshold": similarity_threshold,
            },
        }

    return filtered_boundary_dict


def build_source_fine_boundary_points(
    feature_map: torch.Tensor,
    seg_label: torch.Tensor,
    num_classes: int,
    boundary_kernel: int = 3,
    keep_ratio: float = 0.8,
    min_points: int = 10,
    ignore_background_order: bool = True,
) -> Dict[str, Dict]:
    """
    将“几何候选边界提取 + 细粒度边界归类 + 特征一致性筛选”整合成一个完整入口。

    参数:
        feature_map: torch.Tensor
            特征图，形状为 `(B, C, H, W)`。
        seg_label: torch.Tensor
            源域真实标签图，形状为 `(B, H, W)` 或 `(H, W)`。
        num_classes: int
            总类别数，包含背景 0。
        boundary_kernel: int
            形态学边界提取和局部邻域统计使用的核大小。
        keep_ratio: float
            特征一致性筛选的保留比例。
        min_points: int
            特征一致性筛选的最小点数阈值。
        ignore_background_order: bool
            是否把背景边界固定表示为 `(A, 0)`。

    返回:
        result_dict: dict
            包含三个核心字段：
            - `filtered_boundary_dict`
            - `raw_boundary_dict`
            - `summary_info`
    """
    if feature_map.ndim != 4:
        raise ValueError(f"feature_map 必须是 4 维张量，当前 shape={tuple(feature_map.shape)}")

    seg_label = _ensure_label_is_bhw(seg_label)
    if feature_map.shape[0] != seg_label.shape[0]:
        raise ValueError(
            f"feature_map 与 seg_label 的 batch 大小不一致，"
            f"feature_map.shape[0]={feature_map.shape[0]}, seg_label.shape[0]={seg_label.shape[0]}"
        )

    # 这里主动把 seg_label 放到 feature_map 同一设备上，后续 gather 特征和坐标统计会省心很多。
    seg_label = seg_label.to(feature_map.device)

    raw_boundary_dict = assign_fine_boundary_labels(
        seg_label=seg_label,
        num_classes=num_classes,
        kernel_size=boundary_kernel,
        ignore_background_order=ignore_background_order,
    )

    filtered_boundary_dict = filter_boundary_points_by_feature_consistency(
        boundary_dict=raw_boundary_dict,
        feature_map=feature_map,
        keep_ratio=keep_ratio,
        min_points=min_points,
    )

    summary_info = {}
    all_pair_keys = sorted(raw_boundary_dict.keys())
    for pair_key in all_pair_keys:
        raw_count = int(raw_boundary_dict[pair_key]["raw_count"])
        filtered_info = filtered_boundary_dict.get(pair_key, None)
        kept_count = int(filtered_info["kept_count"]) if filtered_info is not None else 0
        similarity_stats = filtered_info["similarity_stats"] if filtered_info is not None else {}

        summary_info[pair_key] = {
            "raw_count": raw_count,
            "kept_count": kept_count,
            "removed_count": raw_count - kept_count,
            "skipped_filtering": bool(filtered_info["skipped_filtering"]) if filtered_info is not None else False,
            "similarity_stats": similarity_stats,
        }

    return {
        "filtered_boundary_dict": filtered_boundary_dict,
        "raw_boundary_dict": raw_boundary_dict,
        "summary_info": summary_info,
    }


def visualize_boundary_points_on_label(
    seg_label_2d: TensorLike,
    boundary_points_dict: Dict[BoundaryKey, Dict[str, torch.Tensor]],
    save_path: Union[str, Path],
    target_batch_idx: int = 0,
    point_size: int = 10,
    dpi: int = 200,
) -> None:
    """
    将不同细粒度边界类别的点可视化叠加到标签图上，并保存结果图片。

    参数:
        seg_label_2d: TensorLike
            单张 2D 标签图，形状为 `(H, W)`。
        boundary_points_dict: dict
            边界点字典，通常传 `filtered_boundary_dict` 或 `raw_boundary_dict`。
        save_path: str 或 Path
            可视化保存路径。
        target_batch_idx: int
            若输入字典中包含多 batch 坐标，则只可视化该 batch 的点。
        point_size: int
            散点尺寸。
        dpi: int
            图像保存分辨率。
    """
    if isinstance(seg_label_2d, torch.Tensor):
        seg_label_np = seg_label_2d.detach().cpu().numpy()
    else:
        seg_label_np = np.asarray(seg_label_2d)

    if seg_label_np.ndim != 2:
        raise ValueError(f"seg_label_2d 必须是二维数组，当前 shape={seg_label_np.shape}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(seg_label_np, cmap="tab20", interpolation="nearest", alpha=0.75)

    pair_keys = sorted(boundary_points_dict.keys())
    point_cmap = plt.get_cmap("gist_ncar", max(len(pair_keys), 1))

    for idx, pair_key in enumerate(pair_keys):
        info = boundary_points_dict[pair_key]
        coords = info["coords"].detach().cpu()
        if coords.numel() == 0:
            continue

        # 这里只显示指定 batch 的点，避免多张图混在一起看得头疼。
        batch_mask = coords[:, 0] == target_batch_idx
        coords_batch = coords[batch_mask]
        if coords_batch.numel() == 0:
            continue

        y = coords_batch[:, 1].numpy()
        x = coords_batch[:, 2].numpy()
        ax.scatter(
            x,
            y,
            s=point_size,
            c=[point_cmap(idx)],
            label=str(pair_key),
            marker="o",
            edgecolors="none",
        )

    ax.set_title("Fine Boundary Points Overlay")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8, ncol=1)
    ax.set_xlim(0, seg_label_np.shape[1] - 1)
    ax.set_ylim(seg_label_np.shape[0] - 1, 0)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _create_demo_seg_label(height: int = 128, width: int = 128) -> torch.Tensor:
    """
    构造一个最小可运行 demo 用的二维标签图。

    标签设置:
        - 0: 背景
        - 1: 左上矩形器官
        - 2: 与类别 1 直接贴边的右上矩形器官
        - 3: 左下椭圆器官

    参数:
        height: int
            图像高度 H。
        width: int
            图像宽度 W。

    返回:
        seg_label: torch.Tensor
            形状为 `(1, H, W)` 的 long 张量。
    """
    seg_label = torch.zeros((1, height, width), dtype=torch.long)
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )

    # 类别 1：左上矩形
    organ1_mask = (yy >= 22) & (yy <= 76) & (xx >= 20) & (xx <= 57)
    seg_label[0][organ1_mask] = 1

    # 类别 2：右上矩形，和类别 1 在 x=57/58 处直接接壤，没有背景隔开。
    organ2_mask = (yy >= 28) & (yy <= 80) & (xx >= 58) & (xx <= 98)
    seg_label[0][organ2_mask] = 2

    # 类别 3：左下椭圆
    ellipse_mask = ((yy - 92.0) / 16.0) ** 2 + ((xx - 46.0) / 24.0) ** 2 <= 1.0
    seg_label[0][ellipse_mask] = 3

    return seg_label


def _create_demo_feature_map(seg_label: torch.Tensor, channels: int = 16) -> torch.Tensor:
    """
    为 demo 构造一个具有空间结构和类别偏置的特征图。

    设计思路:
        1. 用坐标场构造一组平滑的空间特征，保证边界附近特征不是纯随机噪声
        2. 再叠加一个较弱的类别偏置，让不同器官区域的特征分布既有差别，又不至于完全分裂
        3. 最后加少量噪声，模拟训练中真实特征图的不完美状态

    参数:
        seg_label: torch.Tensor
            标签图，形状为 `(1, H, W)`。
        channels: int
            特征通道数 C。

    返回:
        feature_map: torch.Tensor
            构造好的特征图，形状为 `(1, C, H, W)`。
    """
    seg_label = _ensure_label_is_bhw(seg_label)
    batch_size, height, width = seg_label.shape
    if batch_size != 1:
        raise ValueError("demo 生成器当前只支持 batch_size=1")

    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, steps=height),
        torch.linspace(0.0, 1.0, steps=width),
        indexing="ij",
    )

    # 先构造一组平滑的坐标特征，这些分量在空间上连续，适合拿来模拟编码特征。
    spatial_basis = [
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
        torch.cos(2 * math.pi * (yy + xx)),
        torch.sin(3 * math.pi * yy),
        torch.cos(3 * math.pi * xx),
        torch.sin(3 * math.pi * (yy - xx)),
    ]

    if channels > len(spatial_basis):
        raise ValueError(f"demo channels 目前最多支持 {len(spatial_basis)}，当前请求 {channels}")

    spatial_feature = torch.stack(spatial_basis[:channels], dim=0).unsqueeze(0)

    # 类别偏置的幅度故意设得不大，免得同一条边界两侧的点特征差太开，单原型会很难受。
    class_prototypes = torch.tensor(
        [
            [0.00, 0.05, -0.02, 0.03, 0.00, 0.02, -0.01, 0.00, 0.01, 0.00, 0.02, -0.01, 0.00, 0.01, 0.00, -0.02],
            [0.08, 0.02, 0.03, -0.01, 0.05, 0.01, 0.00, 0.02, -0.02, 0.01, 0.00, 0.02, 0.01, -0.01, 0.00, 0.01],
            [-0.03, 0.07, 0.02, 0.01, -0.02, 0.04, 0.01, -0.01, 0.02, 0.00, 0.03, 0.00, -0.02, 0.02, 0.01, 0.00],
            [0.02, -0.04, 0.05, 0.03, 0.01, -0.02, 0.04, 0.00, -0.01, 0.03, 0.00, 0.01, 0.02, 0.00, -0.02, 0.03],
        ],
        dtype=torch.float32,
    )[:, :channels]

    class_bias = class_prototypes[seg_label[0]].permute(2, 0, 1).unsqueeze(0)

    # 加一点随机噪声，模拟真实训练时特征图本来就不会特别规整。
    noise = 0.03 * torch.randn((1, channels, height, width), dtype=torch.float32)

    feature_map = spatial_feature + 0.35 * class_bias + noise
    return feature_map


def _print_demo_summary(summary_info: Dict[BoundaryKey, Dict[str, Union[int, bool, Dict[str, float]]]]) -> None:
    """
    以更易读的方式打印 demo 里的边界筛选统计结果。

    参数:
        summary_info: dict
            `build_source_fine_boundary_points` 返回的 `summary_info`。
    """
    print("=" * 80)
    print("细粒度边界点筛选统计")
    print("=" * 80)
    if len(summary_info) == 0:
        print("没有提取到任何边界点。")
        return

    for pair_key in sorted(summary_info.keys()):
        info = summary_info[pair_key]
        print(
            f"边界类别 {pair_key}: "
            f"raw_count={info['raw_count']}, "
            f"kept_count={info['kept_count']}, "
            f"removed_count={info['removed_count']}, "
            f"skipped_filtering={info['skipped_filtering']}"
        )


def main() -> None:
    """
    最小可运行 demo。

    demo 内容:
        1. 构造一个包含背景和 3 个前景类别的二维标签图
        2. 构造一个与标签对齐的模拟特征图
        3. 跑通细粒度边界点提取与特征一致性筛选
        4. 打印每个边界类别筛选前后的数量
        5. 保存一张可视化图像
    """
    torch.manual_seed(42)
    np.random.seed(42)

    seg_label = _create_demo_seg_label(height=128, width=128)
    feature_map = _create_demo_feature_map(seg_label=seg_label, channels=16)

    results = build_source_fine_boundary_points(
        feature_map=feature_map,
        seg_label=seg_label,
        num_classes=4,
        boundary_kernel=3,
        keep_ratio=0.8,
        min_points=10,
        ignore_background_order=True,
    )

    _print_demo_summary(results["summary_info"])

    save_path = Path(__file__).resolve().parent / "demo_source_fine_boundary_points.png"
    visualize_boundary_points_on_label(
        seg_label_2d=seg_label[0],
        boundary_points_dict=results["filtered_boundary_dict"],
        save_path=save_path,
        target_batch_idx=0,
        point_size=12,
        dpi=200,
    )

    print("-" * 80)
    print(f"可视化结果已保存到: {save_path}")
    print("-" * 80)


if __name__ == "__main__":
    main()
