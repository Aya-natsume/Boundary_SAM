"""源域细粒度边界点提取与净化模块。

本文件只处理“第一部分”：
1. 从源域真实标签中提取候选边界带。
2. 在候选边界带内部做局部邻接判定，赋予细粒度边界类别。
3. 基于特征一致性做轻量净化，保留高质量边界点。
4. 输出按边界类别组织的点坐标、点特征与统计信息，并提供可视化与 demo。

注意：
1. 当前文件故意不实现 prototype library、目标域分数、SAM prompt、SAM refinement 等后续模块。
2. 当前净化策略只使用“单原型 + keep_ratio 截断”的轻量方式，先把基础打稳，别急着把事情搞复杂。
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, MutableMapping, Optional, Tuple

import torch
import torch.nn.functional as F


BoundaryKey = Tuple[int, int]
BoundaryDict = Dict[BoundaryKey, Dict[str, torch.Tensor | int | bool | float]]


def canonicalize_ordered_boundary_key(a: int, b: int) -> BoundaryKey:
    """规范化有序细粒度边界类别 key。

    输入：
        a: int
            当前边界点所属类别编号。
        b: int
            当前边界点局部邻域中接触到的类别编号。

    输出：
        boundary_key: Tuple[int, int]
            规范化后的有序边界类别 key，始终保持为 (a, b)。

    说明：
        1. 当前项目现在只允许“有序边界”定义。
        2. (A, B) 与 (B, A) 必须视为两个不同类别，绝不交换、绝不合并。
        3. 背景边界同样保持顺序，例如：
           - (1, 0) 表示“类别 1 一侧接触背景”
           - (0, 1) 若未来有需求，也仍然会被保序表示成 (0, 1)
    """
    return int(a), int(b)


def _prepare_binary_mask(binary_mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """把二值 mask 规范成 4D 张量，便于调用池化算子。

    输入：
        binary_mask: Tensor
            形状可为：
            - (H, W)
            - (1, H, W)
            - (B, H, W)
            - (B, 1, H, W)

    输出：
        mask_4d: Tensor, shape = (N, 1, H, W)
            规范后的 4D float 张量。
        original_dim: int
            输入张量的原始维度，用于后面恢复形状。
    """
    original_dim = binary_mask.dim()
    if original_dim == 2:
        mask_4d = binary_mask.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:
        mask_4d = binary_mask.unsqueeze(1)
    elif original_dim == 4:
        if binary_mask.size(1) != 1:
            raise ValueError("binary_mask with 4 dims must have shape (N, 1, H, W)")
        mask_4d = binary_mask
    else:
        raise ValueError("binary_mask must have shape (H, W), (1, H, W), (B, H, W), or (B, 1, H, W)")
    mask_4d = (mask_4d > 0).to(dtype=torch.float32)
    return mask_4d, original_dim


def _restore_boundary_shape(boundary_mask: torch.Tensor, original_dim: int) -> torch.Tensor:
    """把 4D 边界张量恢复回更贴近输入的形状。"""
    if original_dim == 2:
        return boundary_mask[0, 0]
    if original_dim == 3:
        return boundary_mask[:, 0]
    return boundary_mask


def extract_morph_boundary(
    binary_mask: torch.Tensor,
    kernel_size: int = 3,
    mode: str = "inner",
) -> torch.Tensor:
    """使用形态学方法提取边界带。

    输入：
        binary_mask: Tensor
            形状可为：
            - (H, W)
            - (1, H, W)
            - (B, H, W)
            - (B, 1, H, W)
            数值语义为二值 mask，非零表示前景。
        kernel_size: int
            形态学核大小，建议使用奇数，默认 3。
        mode: str
            边界模式。
            - "inner": inner boundary = mask - erode(mask)
            - "gradient": gradient boundary = dilate(mask) - erode(mask)

    输出：
        boundary_mask: Tensor
            与输入相匹配的边界布尔 mask。
            例如输入是 (H, W)，输出就是 (H, W)。
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if mode not in {"inner", "gradient"}:
        raise ValueError("mode must be either 'inner' or 'gradient'")

    mask_4d, original_dim = _prepare_binary_mask(binary_mask)
    padding = kernel_size // 2

    dilated = F.max_pool2d(mask_4d, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = 1.0 - F.max_pool2d(1.0 - mask_4d, kernel_size=kernel_size, stride=1, padding=padding)

    if mode == "inner":
        boundary_4d = (mask_4d > 0.5) & (eroded < 0.5)
    else:
        boundary_4d = (dilated - eroded) > 0.0

    return _restore_boundary_shape(boundary_4d, original_dim)


def _build_local_class_count_map(seg_label: torch.Tensor, num_classes: int, kernel_size: int) -> torch.Tensor:
    """构建每个像素邻域内的类别计数图。

    输入：
        seg_label: Tensor, shape = (B, H, W)
            真实标签图。
        num_classes: int
            类别总数，包含背景 0。
        kernel_size: int
            邻域窗口大小。

    输出：
        local_counts: Tensor, shape = (B, num_classes, H, W)
            local_counts[b, k, y, x] 表示以 (y, x) 为中心的局部窗口内，类别 k 出现了多少个像素。
    """
    one_hot = F.one_hot(seg_label.long(), num_classes=num_classes).permute(0, 3, 1, 2).to(torch.float32)
    kernel = torch.ones(
        num_classes,
        1,
        kernel_size,
        kernel_size,
        device=seg_label.device,
        dtype=torch.float32,
    )
    padding = kernel_size // 2
    local_counts = F.conv2d(one_hot, kernel, padding=padding, groups=num_classes)
    return local_counts


def _stack_boundary_coord_lists(
    boundary_coord_lists: MutableMapping[BoundaryKey, List[torch.Tensor]],
) -> BoundaryDict:
    """把按列表暂存的坐标整理成正式字典。"""
    boundary_dict: BoundaryDict = {}
    for boundary_key, coord_list in boundary_coord_lists.items():
        if len(coord_list) == 0:
            continue
        coords = torch.cat(coord_list, dim=0)
        boundary_dict[boundary_key] = {
            "coords": coords,
            "raw_count": int(coords.size(0)),
        }
    return boundary_dict


def assign_fine_boundary_labels(
    seg_label: torch.Tensor,
    num_classes: int,
    boundary_kernel: int = 3,
    neighbor_kernel: Optional[int] = None,
) -> BoundaryDict:
    """为候选边界点赋予细粒度边界类别。

    输入：
        seg_label: Tensor, shape = (B, H, W)
            源域真实标签图。
        num_classes: int
            类别总数，包含背景 0。
        boundary_kernel: int
            形态学边界带提取时使用的核大小。
        neighbor_kernel: Optional[int]
            局部邻接判定时使用的邻域大小。
            - 当为 None 时，默认与 boundary_kernel 相同。
    输出：
        boundary_dict: dict
            形式示例：
            {
                (A, B): {
                    "coords": Tensor[N, 3],
                    "raw_count": int,
                }
            }

    规则说明：
        1. 先对每个类别构造二值 mask，这里包含背景 0。
        2. 再用形态学 inner boundary 提取候选边界带。
        3. 只在候选边界带内部做局部邻接判定，不做全图逐像素硬扫。
        4. 如果邻域内同时出现多个前景类别，则选择邻接像素数量最多的类别。
        5. 如果多个前景类别并列最多，则 torch.max 会稳定地取编号更小的那个，规则固定，不耍花样。
        6. 当前函数输出的是“有序边界”：
           - 当前点属于 A，且邻域接触 B，则记为 (A, B)。
           - 同一条 A/B 几何边界，会同时拆成 (A, B) 与 (B, A) 两个集合。
        7. 现在器官-背景边界会显式输出双向两侧：
           - (A, 0): 器官 A 一侧接触背景
           - (0, A): 背景一侧接触器官 A
        8. 对任意当前类别 X，若邻域内同时出现前景类别与背景，则仍然优先分到“接触前景”的有序边界类别。
    """
    if seg_label.dim() != 3:
        raise ValueError("seg_label must have shape (B, H, W)")
    if boundary_kernel < 1 or boundary_kernel % 2 == 0:
        raise ValueError("boundary_kernel must be a positive odd integer")
    if neighbor_kernel is None:
        neighbor_kernel = boundary_kernel
    if neighbor_kernel < 1 or neighbor_kernel % 2 == 0:
        raise ValueError("neighbor_kernel must be a positive odd integer")

    seg_label = seg_label.long()
    batch_size, height, width = seg_label.shape
    if num_classes <= 1:
        raise ValueError("num_classes must be larger than 1")

    local_counts = _build_local_class_count_map(seg_label, num_classes=num_classes, kernel_size=neighbor_kernel)
    local_counts_hwk = local_counts.permute(0, 2, 3, 1).contiguous()
    boundary_coord_lists: DefaultDict[BoundaryKey, List[torch.Tensor]] = defaultdict(list)

    for organ_class in range(num_classes):
        organ_mask = seg_label == organ_class
        if not organ_mask.any():
            continue

        candidate_boundary = extract_morph_boundary(organ_mask, kernel_size=boundary_kernel, mode="inner")
        candidate_coords = torch.nonzero(candidate_boundary, as_tuple=False)
        if candidate_coords.numel() == 0:
            continue

        neighborhood_counts = local_counts_hwk[
            candidate_coords[:, 0],
            candidate_coords[:, 1],
            candidate_coords[:, 2],
        ]

        neighbor_counts = neighborhood_counts.clone()
        neighbor_counts[:, organ_class] = 0.0

        foreground_neighbor_counts = neighbor_counts.clone()
        foreground_neighbor_counts[:, 0] = 0.0

        max_foreground_count, best_foreground_class = foreground_neighbor_counts.max(dim=1)
        background_count = neighbor_counts[:, 0]

        has_foreground_neighbor = max_foreground_count > 0
        has_background_neighbor = background_count > 0
        valid_mask = has_foreground_neighbor | has_background_neighbor
        if not valid_mask.any():
            continue

        assigned_neighbor_class = torch.where(
            has_foreground_neighbor,
            best_foreground_class,
            torch.zeros_like(best_foreground_class),
        )








        valid_coords = candidate_coords[valid_mask]
        valid_neighbor_class = assigned_neighbor_class[valid_mask]

        unique_neighbor_classes = torch.unique(valid_neighbor_class)
        for neighbor_class in unique_neighbor_classes.tolist():
            class_mask = valid_neighbor_class == neighbor_class
            coords_this_key = valid_coords[class_mask]
            boundary_key = canonicalize_ordered_boundary_key(organ_class, neighbor_class)
            boundary_coord_lists[boundary_key].append(coords_this_key)

    return _stack_boundary_coord_lists(boundary_coord_lists)


def gather_point_features(feature_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """根据坐标一次性提取点特征。

    输入：
        feature_map: Tensor, shape = (B, C, H, W)
            源域特征图。
        coords: Tensor, shape = (N, 3)
            每行为 [batch_idx, y, x]。

    输出：
        point_features: Tensor, shape = (N, C)
            与输入坐标一一对应的点特征。
    """
    if feature_map.dim() != 4:
        raise ValueError("feature_map must have shape (B, C, H, W)")
    if coords.dim() != 2 or coords.size(1) != 3:
        raise ValueError("coords must have shape (N, 3)")

    batch_size, channels, height, width = feature_map.shape
    if coords.numel() == 0:
        return feature_map.new_zeros((0, channels))

    coords = coords.long()
    batch_idx = coords[:, 0]
    y_idx = coords[:, 1]
    x_idx = coords[:, 2]

    if batch_idx.min() < 0 or batch_idx.max() >= batch_size:
        raise IndexError("coords batch indices are out of range")
    if y_idx.min() < 0 or y_idx.max() >= height or x_idx.min() < 0 or x_idx.max() >= width:
        raise IndexError("coords spatial indices are out of range")

    point_features = feature_map[batch_idx, :, y_idx, x_idx]
    return point_features


def filter_boundary_points_by_feature_consistency(
    boundary_dict: BoundaryDict,
    feature_map: torch.Tensor,
    keep_ratio: float = 0.8,
    min_points: int = 10,
) -> BoundaryDict:
    """基于特征一致性净化边界点。

    输入：
        boundary_dict: dict
            assign_fine_boundary_labels 的输出字典。
        feature_map: Tensor, shape = (B, C, H, W)
            当前网络某层特征图。
        keep_ratio: float
            保留比例，例如 0.8 表示只保留相似度最高的 80%。
        min_points: int
            最小筛选点数阈值。
            当某一边界类别的点数小于该阈值时，默认不筛选，全部保留。
    输出：
        filtered_boundary_dict: dict
            形式示例：
            {
                (A, B): {
                    "coords": Tensor[M, 3],
                    "features": Tensor[M, C],
                    "raw_count": int,
                    "kept_count": int,
                    "similarity": Tensor[M],
                    "prototype": Tensor[C],
                    "skip_filter": bool,
                    "actual_keep_ratio": float,
                }
            }

    说明：
        1. 当前阶段只做“单原型 + keep_ratio 截断”，不做聚类，不做多子原型。
        2. 原型直接由同类候选点的 L2 归一化特征均值构成，简单，但够快。
    """
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in the range (0, 1]")
    if min_points < 1:
        raise ValueError("min_points must be >= 1")

    filtered_boundary_dict: BoundaryDict = {}

    for raw_key, raw_data in boundary_dict.items():
        boundary_key = canonicalize_ordered_boundary_key(raw_key[0], raw_key[1])
        coords = raw_data["coords"]
        if not isinstance(coords, torch.Tensor):
            raise TypeError("boundary_dict['coords'] must be a torch.Tensor")

        raw_count = int(coords.size(0))
        point_features = gather_point_features(feature_map, coords)

        if raw_count == 0:
            filtered_boundary_dict[boundary_key] = {
                "coords": coords,
                "features": point_features,
                "raw_count": 0,
                "kept_count": 0,
                "similarity": feature_map.new_zeros((0,)),
                "prototype": feature_map.new_zeros((feature_map.size(1),)),
                "skip_filter": True,
                "actual_keep_ratio": 0.0,
            }
            continue

        normalized_features = F.normalize(point_features, p=2, dim=1, eps=1e-12)
        prototype = normalized_features.mean(dim=0)
        prototype = F.normalize(prototype.unsqueeze(0), p=2, dim=1, eps=1e-12).squeeze(0)
        similarity = torch.matmul(normalized_features, prototype)

        skip_filter = raw_count < min_points or keep_ratio >= 1.0
        if skip_filter:
            keep_indices = torch.arange(raw_count, device=coords.device)
        else:
            keep_count = max(1, math.ceil(raw_count * keep_ratio))
            keep_indices = torch.topk(similarity, k=keep_count, largest=True).indices
            keep_indices = keep_indices[torch.argsort(keep_indices)]

        kept_coords = coords[keep_indices]
        kept_features = point_features[keep_indices]
        kept_similarity = similarity[keep_indices]
        kept_count = int(kept_coords.size(0))
        actual_keep_ratio = float(kept_count / raw_count) if raw_count > 0 else 0.0

        filtered_boundary_dict[boundary_key] = {
            "coords": kept_coords,
            "features": kept_features,
            "raw_count": raw_count,
            "kept_count": kept_count,
            "similarity": kept_similarity,
            "prototype": prototype,
            "skip_filter": bool(skip_filter),
            "actual_keep_ratio": actual_keep_ratio,
        }

    return filtered_boundary_dict


def build_source_fine_boundary_points(
    feature_map: torch.Tensor,
    seg_label: torch.Tensor,
    num_classes: int,
    boundary_kernel: int = 3,
    neighbor_kernel: Optional[int] = None,
    keep_ratio: float = 0.8,
    min_points: int = 10,
) -> Tuple[BoundaryDict, BoundaryDict, Dict[BoundaryKey, Dict[str, int | float | bool]]]:
    """构建源域高质量细粒度边界点集。

    输入：
        feature_map: Tensor, shape = (B, C, H, W)
            当前网络某层特征图。
        seg_label: Tensor, shape = (B, H, W)
            源域真实标签图。
        num_classes: int
            类别总数，包含背景 0。
        boundary_kernel: int
            形态学边界带提取时使用的核大小。
        neighbor_kernel: Optional[int]
            局部邻接判定时使用的核大小。
            - 当为 None 时，默认与 boundary_kernel 相同。
        keep_ratio: float
            特征一致性净化时的保留比例。
        min_points: int
            低于该点数的边界类别默认跳过筛选。
    输出：
        raw_boundary_dict: dict
            细粒度边界候选点字典，主要包含 coords 与 raw_count。
        filtered_boundary_dict: dict
            特征一致性净化后的高质量边界点字典。
        summary_info: dict
            每个边界类别的摘要统计信息。
    """
    if feature_map.dim() != 4:
        raise ValueError("feature_map must have shape (B, C, H, W)")
    if seg_label.dim() != 3:
        raise ValueError("seg_label must have shape (B, H, W)")
    if feature_map.size(0) != seg_label.size(0) or feature_map.size(2) != seg_label.size(1) or feature_map.size(3) != seg_label.size(2):
        raise ValueError("feature_map and seg_label must share the same batch and spatial size")

    raw_boundary_dict = assign_fine_boundary_labels(
        seg_label=seg_label,
        num_classes=num_classes,
        boundary_kernel=boundary_kernel,
        neighbor_kernel=neighbor_kernel,
    )

    filtered_boundary_dict = filter_boundary_points_by_feature_consistency(
        boundary_dict=raw_boundary_dict,
        feature_map=feature_map,
        keep_ratio=keep_ratio,
        min_points=min_points,
    )

    summary_info: Dict[BoundaryKey, Dict[str, int | float | bool]] = {}
    all_keys = sorted(filtered_boundary_dict.keys())
    for boundary_key in all_keys:
        raw_data = raw_boundary_dict[boundary_key]
        filtered_data = filtered_boundary_dict[boundary_key]
        raw_count = int(raw_data["raw_count"])
        kept_count = int(filtered_data["kept_count"])
        summary_info[boundary_key] = {
            "raw_count": raw_count,
            "kept_count": kept_count,
            "actual_keep_ratio": float(filtered_data["actual_keep_ratio"]),
            "skip_filter": bool(filtered_data["skip_filter"]),
        }

    return raw_boundary_dict, filtered_boundary_dict, summary_info


def _boundary_key_to_string(boundary_key: BoundaryKey) -> str:
    """把边界 key 转成便于显示的字符串。"""
    return f"({boundary_key[0]}, {boundary_key[1]})"


def visualize_fine_boundary_points(
    seg_label_2d: torch.Tensor,
    boundary_dict: BoundaryDict,
    save_path: str | Path,
) -> None:
    """可视化单张标签图上的细粒度边界点。

    输入：
        seg_label_2d: Tensor, shape = (H, W) 或 (1, H, W)
            单张标签图。
        boundary_dict: dict
            细粒度边界点字典。
            函数会默认可视化 batch_idx=0 的点。
        save_path: str | Path
            图像保存路径。

    输出：
        无返回值。
        图像会直接保存到磁盘。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if seg_label_2d.dim() == 3:
        if seg_label_2d.size(0) != 1:
            raise ValueError("seg_label_2d with 3 dims must have shape (1, H, W)")
        seg_label_2d = seg_label_2d[0]
    if seg_label_2d.dim() != 2:
        raise ValueError("seg_label_2d must have shape (H, W) or (1, H, W)")

    seg_np = seg_label_2d.detach().cpu().numpy()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_keys = sorted(boundary_dict.keys())
    cmap = plt.get_cmap("tab20", max(1, len(sorted_keys)))

    plt.figure(figsize=(8, 8))
    plt.imshow(seg_np, cmap="nipy_spectral", interpolation="nearest", alpha=0.85)

    legend_handles = []
    legend_labels = []
    total_points = 0

    for color_index, boundary_key in enumerate(sorted_keys):
        data = boundary_dict[boundary_key]
        coords = data["coords"]
        if not isinstance(coords, torch.Tensor) or coords.numel() == 0:
            continue
        coords = coords.detach().cpu()
        batch_mask = coords[:, 0] == 0
        coords_2d = coords[batch_mask]
        if coords_2d.numel() == 0:
            continue

        ys = coords_2d[:, 1].numpy()
        xs = coords_2d[:, 2].numpy()
        scatter = plt.scatter(
            xs,
            ys,
            s=10,
            c=[cmap(color_index)],
            marker="o",
            linewidths=0.0,
            alpha=0.95,
        )
        legend_handles.append(scatter)
        legend_labels.append(f"{_boundary_key_to_string(boundary_key)}: {coords_2d.size(0)}")
        total_points += int(coords_2d.size(0))

    plt.title(f"Fine Boundary Points (batch=0, total={total_points})")
    plt.axis("off")
    if legend_handles:
        plt.legend(legend_handles, legend_labels, loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def _create_dummy_seg_label(height: int = 96, width: int = 96) -> torch.Tensor:
    """构造一个最小可运行的 dummy 标签图。

    输出：
        seg_label: Tensor, shape = (1, H, W)
            其中包含背景 0 和 3 个前景类别。
    """
    seg_label = torch.zeros((1, height, width), dtype=torch.long)
    seg_label[0, 12:48, 10:34] = 1
    seg_label[0, 18:58, 34:64] = 2
    seg_label[0, 54:84, 16:42] = 3

    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    circle_mask = (yy - 70) ** 2 + (xx - 70) ** 2 <= 9 ** 2
    seg_label[0, circle_mask] = 2
    return seg_label


def _create_dummy_feature_map(seg_label: torch.Tensor, channels: int = 8) -> torch.Tensor:
    """根据 dummy 标签图构造一个可运行的 dummy 特征图。

    输入：
        seg_label: Tensor, shape = (B, H, W)
            dummy 标签图。
        channels: int
            特征通道数。

    输出：
        feature_map: Tensor, shape = (B, C, H, W)
            与标签图空间尺寸一致的模拟特征图。
    """
    torch.manual_seed(7)
    batch_size, height, width = seg_label.shape
    num_classes = int(seg_label.max().item()) + 1

    class_prototypes = torch.randn(num_classes, channels, dtype=torch.float32)
    class_prototypes[0] = torch.zeros(channels, dtype=torch.float32)
    class_prototypes = F.normalize(class_prototypes, dim=1) * 2.0

    base_feature = class_prototypes[seg_label]
    base_feature = base_feature + 0.15 * torch.randn_like(base_feature)

    yy = torch.linspace(-1.0, 1.0, steps=height).view(1, height, 1, 1)
    xx = torch.linspace(-1.0, 1.0, steps=width).view(1, 1, width, 1)
    base_feature[..., 0:1] = base_feature[..., 0:1] + 0.30 * yy
    base_feature[..., 1:2] = base_feature[..., 1:2] + 0.30 * xx

    corruption_mask = torch.zeros((batch_size, height, width), dtype=torch.bool)
    corruption_mask[:, 28:42, 30:44] = True
    base_feature[corruption_mask] = torch.randn_like(base_feature[corruption_mask]) * 2.5

    feature_map = base_feature.permute(0, 3, 1, 2).contiguous()
    return feature_map


def main() -> None:
    """最小可运行 demo。"""
    seg_label = _create_dummy_seg_label()
    feature_map = _create_dummy_feature_map(seg_label=seg_label, channels=8)
    num_classes = 4

    raw_boundary_dict, filtered_boundary_dict, summary_info = build_source_fine_boundary_points(
        feature_map=feature_map,
        seg_label=seg_label,
        num_classes=num_classes,
        boundary_kernel=3,
        neighbor_kernel=3,
        keep_ratio=0.8,
        min_points=10,
    )

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_vis_path = output_dir / "fine_boundary_points_raw.png"
    filtered_vis_path = output_dir / "fine_boundary_points_filtered.png"

    visualize_fine_boundary_points(seg_label_2d=seg_label[0], boundary_dict=raw_boundary_dict, save_path=raw_vis_path)
    visualize_fine_boundary_points(seg_label_2d=seg_label[0], boundary_dict=filtered_boundary_dict, save_path=filtered_vis_path)

    print("=" * 80)
    print("Source Fine Boundary Point Demo")
    print("=" * 80)
    for boundary_key in sorted(summary_info.keys()):
        stats = summary_info[boundary_key]
        print(
            f"boundary={_boundary_key_to_string(boundary_key):>8s} | "
            f"raw={int(stats['raw_count']):4d} | "
            f"kept={int(stats['kept_count']):4d} | "
            f"actual_keep_ratio={float(stats['actual_keep_ratio']):.3f} | "
            f"skip_filter={bool(stats['skip_filter'])}"
        )
    print("-" * 80)
    print(f"raw visualization saved to:      {raw_vis_path}")
    print(f"filtered visualization saved to: {filtered_vis_path}")



if __name__ == "__main__":
    main()
