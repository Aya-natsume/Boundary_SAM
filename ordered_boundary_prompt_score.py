"""第三部分：从粗边界与 ordered prototype 出发构造局部条带框。

当前新版第三部分只做下面这些事：
1. 在粗边界局部区域中确定一个几何中心点 `c`。
2. 在 `c` 附近提取一小段局部粗边界点，并用 PCA 拟合局部切线 `t` 与法线 `n`。
3. 沿法线两侧分别离散采样，计算与 ordered boundary prototype 的余弦相似度曲线。
4. 在两条一维相似度曲线上做平滑与变点检测，得到两侧边界厚度截止点 `q_a` 与 `q_b`。
5. 用 `q_a / q_b` 与切线 `t` 构造一个几何上更合理的局部条带框。

注意：
1. 前两步保持不变，本文件只重写第三步。
2. 当前文件不再把第三步理解成“在 box 里找两个核心点”。
3. 为了兼容仓库已有调用路径，`generate_ordered_core_points_in_box` 等函数名保留，
   但它们现在返回的是“条带框几何结果”；其中 `core_ab / core_ba` 只是兼容字段，
   分别别名到 `q_a / q_b`，便于旧入口继续往下传递。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from dynamic_boundary_prototype_bank import DynamicBoundaryPrototypeBank


BoundaryKey = Tuple[int, int]
BoundaryBox = Dict[str, object]
CorePoint = Dict[str, object]
PrototypeLibrary = (
    DynamicBoundaryPrototypeBank
    | Dict[BoundaryKey, Dict[str, torch.Tensor | int | float]]
    | Dict[BoundaryKey, torch.Tensor]
)


DEFAULT_STRIP_BOX_CONFIG: Dict[str, float | int] = {
    "boundary_response_temperature": 0.20,
    "boundary_min_pair_support": 0.05,
    "boundary_response_quantile": 0.75,
    "center_choose_topk": 48,
    "local_fit_radius": 14.0,
    "local_fit_topk": 96,
    "normal_orientation_probe_distance": 2.5,
    "normal_num_samples": 33,
    "normal_scan_margin": 1.0,
    "curve_smooth_kernel": 5,
    "curve_change_window": 3,
    "curve_min_drop": 0.02,
    "strip_normal_padding": 0.75,
    "strip_tangent_padding": 4.0,
    "strip_min_half_length": 10.0,
    "strip_max_half_length": 28.0,
}


def canonicalize_ordered_boundary_key(a: int, b: int) -> BoundaryKey:
    """规范化有序边界 key。

    输入：
        a: int
            当前侧类别 id。
        b: int
            邻接侧类别 id。

    输出：
        boundary_key: Tuple[int, int]
            固定返回 `(a, b)`，绝不排序、绝不合并。
    """
    return int(a), int(b)


def l2_normalize_feature(x: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    """对输入张量沿指定维度做 L2 normalize。

    输入：
        x: Tensor
            任意形状张量。
        dim: int
            归一化维度。
        eps: float
            数值稳定项。

    输出：
        normalized_x: Tensor
            与输入同形状的归一化张量。
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    return F.normalize(x.to(torch.float32), p=2, dim=dim, eps=eps)


def _parse_box(
    box: BoundaryBox,
    batch_size: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Dict[str, int]:
    """把输入 box 解析成统一格式。

    支持的 box 格式：
        1. `{"batch_idx": b, "box": (x1, y1, x2, y2)}`
        2. `{"batch_idx": b, "x1": ..., "y1": ..., "x2": ..., "y2": ...}`
        3. `{"batch_idx": b, "x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}`

    输出：
        parsed_box: dict
            统一后的闭区间坐标：
            {
                "batch_idx": int,
                "x_min": int,
                "y_min": int,
                "x_max": int,
                "y_max": int,
            }
    """
    if not isinstance(box, dict):
        raise TypeError("box must be a dictionary")

    batch_idx = int(box.get("batch_idx", 0))
    if "box" in box:
        raw_box = box["box"]
        if not isinstance(raw_box, (tuple, list)) or len(raw_box) != 4:
            raise ValueError("box['box'] must be a tuple/list with four elements: (x1, y1, x2, y2)")
        x_min, y_min, x_max, y_max = [int(v) for v in raw_box]
    else:
        x_min = int(box["x_min"]) if "x_min" in box else int(box["x1"])
        y_min = int(box["y_min"]) if "y_min" in box else int(box["y1"])
        x_max = int(box["x_max"]) if "x_max" in box else int(box["x2"])
        y_max = int(box["y_max"]) if "y_max" in box else int(box["y2"])

    if x_max < x_min or y_max < y_min:
        raise ValueError("box must satisfy x_max >= x_min and y_max >= y_min")
    if batch_size is not None and not (0 <= batch_idx < batch_size):
        raise IndexError("box batch_idx is out of range")
    if width is not None and not (0 <= x_min <= x_max < width):
        raise IndexError("box x coordinates are out of range")
    if height is not None and not (0 <= y_min <= y_max < height):
        raise IndexError("box y coordinates are out of range")

    return {
        "batch_idx": batch_idx,
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
    }


def _get_box_size(box_info: Dict[str, int]) -> Tuple[int, int]:
    """返回 box 的 `(h_box, w_box)`。"""
    return int(box_info["y_max"] - box_info["y_min"] + 1), int(box_info["x_max"] - box_info["x_min"] + 1)


def _extract_prototype_from_library(prototype_library: PrototypeLibrary, a: int, b: int) -> Optional[torch.Tensor]:
    """从 ordered prototype library 中取出 `(a, b)` 对应的 prototype。"""
    boundary_key = canonicalize_ordered_boundary_key(a, b)

    if isinstance(prototype_library, DynamicBoundaryPrototypeBank):
        prototype = prototype_library.get(boundary_key[0], boundary_key[1])
        if prototype is None:
            return None
        return prototype.detach().to(torch.float32)

    if not isinstance(prototype_library, dict):
        raise TypeError("prototype_library must be a DynamicBoundaryPrototypeBank or a dictionary")
    if boundary_key not in prototype_library:
        return None

    entry = prototype_library[boundary_key]
    if isinstance(entry, torch.Tensor):
        return entry.detach().to(torch.float32)
    if isinstance(entry, dict):
        prototype = entry.get("prototype")
        if prototype is None:
            return None
        if not isinstance(prototype, torch.Tensor):
            raise TypeError(f"prototype_library[{boundary_key}]['prototype'] must be a torch.Tensor")
        return prototype.detach().to(torch.float32)
    raise TypeError(f"prototype_library[{boundary_key}] must be a Tensor or a dictionary")


def compute_ordered_similarity_in_box(
    feature_map: torch.Tensor,
    box: BoundaryBox,
    prototype: Optional[torch.Tensor],
    normalize_feature: bool = True,
    missing_proto_value: Optional[float] = None,
) -> torch.Tensor:
    """在当前 box 内计算 feature 与 prototype 的余弦相似度图。

    输入：
        feature_map: Tensor, shape = (B, C, H, W)
        box: dict
            当前局部框，采用闭区间坐标。
        prototype: Optional[Tensor], shape = (C,)
            当前 ordered boundary prototype。
        normalize_feature: bool
            是否在当前函数内部对特征做 L2 normalize。
        missing_proto_value: Optional[float]
            prototype 缺失时的填充值。

    输出：
        sim_map_box: Tensor, shape = (h_box, w_box)
            当前 box 内的局部相似度图。
    """
    if feature_map.dim() != 4:
        raise ValueError("feature_map must have shape (B, C, H, W)")

    batch_size, channels, height, width = feature_map.shape
    box_info = _parse_box(box, batch_size=batch_size, height=height, width=width)
    h_box, w_box = _get_box_size(box_info)

    if prototype is None:
        fill_value = 0.0 if missing_proto_value is None else float(missing_proto_value)
        return torch.full((h_box, w_box), fill_value, dtype=torch.float32, device=feature_map.device)

    if prototype.dim() != 1 or prototype.numel() != channels:
        raise ValueError(f"prototype must have shape ({channels},)")

    batch_idx = int(box_info["batch_idx"])
    local_feature = feature_map[
        batch_idx,
        :,
        box_info["y_min"]:box_info["y_max"] + 1,
        box_info["x_min"]:box_info["x_max"] + 1,
    ].to(torch.float32)
    local_feature = local_feature.permute(1, 2, 0).reshape(-1, channels)
    if normalize_feature:
        local_feature = l2_normalize_feature(local_feature, dim=1)

    normalized_prototype = l2_normalize_feature(prototype.to(feature_map.device, dtype=torch.float32), dim=0)
    similarity = torch.matmul(local_feature, normalized_prototype)
    return similarity.reshape(h_box, w_box)


def _crop_pair_probability_maps(prob_map: torch.Tensor, box: BoundaryBox, a: int, b: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """裁出当前 pair 在 box 内的概率图。"""
    if prob_map.dim() != 4:
        raise ValueError("prob_map must have shape (B, K, H, W)")

    batch_size, num_classes, height, width = prob_map.shape
    if not (0 <= int(a) < num_classes and 0 <= int(b) < num_classes):
        raise IndexError("a and b must be valid class indices for prob_map")

    box_info = _parse_box(box, batch_size=batch_size, height=height, width=width)
    batch_idx = int(box_info["batch_idx"])
    pa_box = prob_map[
        batch_idx,
        int(a),
        box_info["y_min"]:box_info["y_max"] + 1,
        box_info["x_min"]:box_info["x_max"] + 1,
    ].to(torch.float32)
    pb_box = prob_map[
        batch_idx,
        int(b),
        box_info["y_min"]:box_info["y_max"] + 1,
        box_info["x_min"]:box_info["x_max"] + 1,
    ].to(torch.float32)
    return pa_box, pb_box, box_info


def _build_boundary_response_from_probabilities(pa_box: torch.Tensor, pb_box: torch.Tensor) -> torch.Tensor:
    """用当前 pair 的两张概率图构造局部边界响应。

    定义：
        pair_support = min(p_a, p_b)
        gap = |p_a - p_b|
        boundary_response = pair_support * exp(-gap / temperature)
    """
    temperature = float(DEFAULT_STRIP_BOX_CONFIG["boundary_response_temperature"])
    pair_support = torch.minimum(pa_box, pb_box)
    gap = torch.abs(pa_box - pb_box)
    boundary_response = pair_support * torch.exp(-gap / max(temperature, 1e-6))
    return boundary_response


def _parse_boundary_coords(
    coarse_boundary_coords: Optional[torch.Tensor],
    batch_idx: int,
    box_info: Dict[str, int],
) -> Optional[torch.Tensor]:
    """把外部给的粗边界坐标解析成 box 内的全局坐标 `(y, x)`。"""
    if coarse_boundary_coords is None:
        return None
    if not isinstance(coarse_boundary_coords, torch.Tensor):
        raise TypeError("coarse_boundary_coords must be a torch.Tensor when provided")
    if coarse_boundary_coords.numel() == 0:
        return None
    if coarse_boundary_coords.dim() != 2 or coarse_boundary_coords.size(1) not in {2, 3}:
        raise ValueError("coarse_boundary_coords must have shape (N, 2) or (N, 3)")

    coords = coarse_boundary_coords.to(torch.float32)
    if coords.size(1) == 3:
        coords = coords[coords[:, 0].long() == int(batch_idx)][:, 1:3]
    if coords.numel() == 0:
        return None

    inside_mask = (
        (coords[:, 0] >= float(box_info["y_min"]))
        & (coords[:, 0] <= float(box_info["y_max"]))
        & (coords[:, 1] >= float(box_info["x_min"]))
        & (coords[:, 1] <= float(box_info["x_max"]))
    )
    coords = coords[inside_mask]
    if coords.numel() == 0:
        return None
    return coords


def extract_local_center_point_from_coarse_boundary(
    prob_map: torch.Tensor,
    box: BoundaryBox,
    a: int,
    b: int,
    coarse_boundary_coords: Optional[torch.Tensor] = None,
) -> Dict[str, float | int]:
    """从当前 pair 的粗边界中提取局部中心点 `c`。

    输入：
        prob_map: Tensor, shape = (B, K, H, W)
            教师模型概率图。
        box: dict
            当前粗定位 box。
        a: int
            当前 pair 的 A 类别。
        b: int
            当前 pair 的 B 类别，可以是背景 0 或另一器官。
        coarse_boundary_coords: Optional[Tensor]
            若上游能提供粗边界坐标，则优先使用。
            支持形状：
            - (N, 3): `[batch_idx, y, x]`
            - (N, 2): `[y, x]`

    输出：
        center_point: dict
            {
                "batch_idx": int,
                "y": float,
                "x": float,
            }

    说明：
        1. 如果 box 里带了粗边界坐标，就优先从这些点里选中心点。
        2. 如果没有，就退化成在 box 内用 pair 概率图构造一个“伪边界响应”，
           再从高响应点里选一个离其加权中心最近的点作为 `c`。
        3. `c` 只是几何参考点，不是最终 prompt。
    """
    pa_box, pb_box, box_info = _crop_pair_probability_maps(prob_map=prob_map, box=box, a=a, b=b)
    batch_idx = int(box_info["batch_idx"])

    coarse_points = _parse_boundary_coords(coarse_boundary_coords, batch_idx=batch_idx, box_info=box_info)
    if coarse_points is not None and coarse_points.size(0) > 0:
        center_estimate = coarse_points.mean(dim=0)
        distances = torch.linalg.norm(coarse_points - center_estimate.unsqueeze(0), dim=1)
        selected_point = coarse_points[int(torch.argmin(distances).item())]
        return {
            "batch_idx": batch_idx,
            "y": float(selected_point[0].item()),
            "x": float(selected_point[1].item()),
        }

    boundary_response = _build_boundary_response_from_probabilities(pa_box=pa_box, pb_box=pb_box)
    pair_support = torch.minimum(pa_box, pb_box)
    min_pair_support = float(DEFAULT_STRIP_BOX_CONFIG["boundary_min_pair_support"])
    candidate_mask = pair_support >= min_pair_support
    candidate_values = boundary_response[candidate_mask]

    if candidate_values.numel() == 0:
        selected_y = 0.5 * float(box_info["y_min"] + box_info["y_max"])
        selected_x = 0.5 * float(box_info["x_min"] + box_info["x_max"])
        return {"batch_idx": batch_idx, "y": selected_y, "x": selected_x}

    quantile = float(DEFAULT_STRIP_BOX_CONFIG["boundary_response_quantile"])
    threshold = float(torch.quantile(candidate_values, q=min(max(quantile, 0.0), 1.0)).item())
    top_mask = candidate_mask & (boundary_response >= threshold)
    top_coords_local = torch.nonzero(top_mask, as_tuple=False)
    top_scores = boundary_response[top_mask]

    if top_coords_local.numel() == 0:
        flat_index = int(torch.argmax(boundary_response.reshape(-1)).item())
        h_box, w_box = boundary_response.shape
        y_local = flat_index // int(w_box)
        x_local = flat_index % int(w_box)
        return {
            "batch_idx": batch_idx,
            "y": float(box_info["y_min"] + y_local),
            "x": float(box_info["x_min"] + x_local),
        }

    topk = min(int(DEFAULT_STRIP_BOX_CONFIG["center_choose_topk"]), int(top_coords_local.size(0)))
    if topk < int(top_coords_local.size(0)):
        top_indices = torch.topk(top_scores, k=topk, largest=True).indices
        top_coords_local = top_coords_local[top_indices]
        top_scores = top_scores[top_indices]

    top_coords_global = torch.stack(
        [top_coords_local[:, 0] + int(box_info["y_min"]), top_coords_local[:, 1] + int(box_info["x_min"])],
        dim=1,
    ).to(torch.float32)
    weighted_center = (top_coords_global * top_scores.unsqueeze(1)).sum(dim=0) / top_scores.sum().clamp_min(1e-6)
    distances = torch.linalg.norm(top_coords_global - weighted_center.unsqueeze(0), dim=1)
    selected_point = top_coords_global[int(torch.argmin(distances).item())]
    return {
        "batch_idx": batch_idx,
        "y": float(selected_point[0].item()),
        "x": float(selected_point[1].item()),
    }


def extract_local_boundary_points_near_center(
    prob_map: torch.Tensor,
    box: BoundaryBox,
    a: int,
    b: int,
    center_point: Dict[str, float | int],
    coarse_boundary_coords: Optional[torch.Tensor] = None,
    fit_radius: Optional[float] = None,
    topk_points: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """在中心点附近提取用于局部直线拟合的一段粗边界点。

    输出：
        boundary_info: dict
            {
                "boundary_response_box": Tensor[h_box, w_box],
                "boundary_points_global": Tensor[N, 2],
                "boundary_points_local": Tensor[N, 2],
                "boundary_scores": Tensor[N],
            }
            其中：
            - `boundary_points_global` 的坐标顺序是 `(y, x)`
            - `boundary_points_local` 的坐标顺序是 `(y_local, x_local)`
    """
    fit_radius = float(DEFAULT_STRIP_BOX_CONFIG["local_fit_radius"]) if fit_radius is None else float(fit_radius)
    topk_points = int(DEFAULT_STRIP_BOX_CONFIG["local_fit_topk"]) if topk_points is None else int(topk_points)

    pa_box, pb_box, box_info = _crop_pair_probability_maps(prob_map=prob_map, box=box, a=a, b=b)
    boundary_response_box = _build_boundary_response_from_probabilities(pa_box=pa_box, pb_box=pb_box)
    batch_idx = int(box_info["batch_idx"])

    coarse_points = _parse_boundary_coords(coarse_boundary_coords, batch_idx=batch_idx, box_info=box_info)
    if coarse_points is None:
        pair_support = torch.minimum(pa_box, pb_box)
        candidate_mask = pair_support >= float(DEFAULT_STRIP_BOX_CONFIG["boundary_min_pair_support"])
        candidate_coords_local = torch.nonzero(candidate_mask, as_tuple=False)
        if candidate_coords_local.numel() == 0:
            candidate_coords_local = torch.nonzero(torch.ones_like(pa_box, dtype=torch.bool), as_tuple=False)
        candidate_coords_global = torch.stack(
            [candidate_coords_local[:, 0] + int(box_info["y_min"]), candidate_coords_local[:, 1] + int(box_info["x_min"])],
            dim=1,
        ).to(torch.float32)
        candidate_scores = boundary_response_box[candidate_coords_local[:, 0], candidate_coords_local[:, 1]]
    else:
        candidate_coords_global = coarse_points.to(torch.float32)
        local_y = (candidate_coords_global[:, 0] - float(box_info["y_min"])).long().clamp_(0, int(pa_box.size(0)) - 1)
        local_x = (candidate_coords_global[:, 1] - float(box_info["x_min"])).long().clamp_(0, int(pa_box.size(1)) - 1)
        candidate_coords_local = torch.stack([local_y, local_x], dim=1)
        candidate_scores = boundary_response_box[local_y, local_x]

    center_tensor = torch.tensor([float(center_point["y"]), float(center_point["x"])], dtype=torch.float32, device=candidate_coords_global.device)
    distances = torch.linalg.norm(candidate_coords_global - center_tensor.unsqueeze(0), dim=1)
    near_mask = distances <= fit_radius
    if near_mask.any():
        candidate_coords_global = candidate_coords_global[near_mask]
        candidate_coords_local = candidate_coords_local[near_mask]
        candidate_scores = candidate_scores[near_mask]
        distances = distances[near_mask]

    if candidate_coords_global.size(0) > topk_points:

        ranking_score = candidate_scores - 0.03 * distances.to(candidate_scores.device)
        keep_indices = torch.topk(ranking_score, k=topk_points, largest=True).indices
        candidate_coords_global = candidate_coords_global[keep_indices]
        candidate_coords_local = candidate_coords_local[keep_indices]
        candidate_scores = candidate_scores[keep_indices]

    if candidate_coords_global.numel() == 0:
        fallback_point = center_tensor.unsqueeze(0)
        fallback_local = torch.tensor(
            [[float(center_point["y"]) - float(box_info["y_min"]), float(center_point["x"]) - float(box_info["x_min"])]],
            dtype=torch.float32,
        )
        return {
            "boundary_response_box": boundary_response_box,
            "boundary_points_global": fallback_point,
            "boundary_points_local": fallback_local,
            "boundary_scores": torch.ones(1, dtype=torch.float32, device=boundary_response_box.device),
        }

    return {
        "boundary_response_box": boundary_response_box,
        "boundary_points_global": candidate_coords_global.to(torch.float32),
        "boundary_points_local": candidate_coords_local.to(torch.float32),
        "boundary_scores": candidate_scores.to(torch.float32),
    }


def _sample_feature_vectors_at_points(feature_map: torch.Tensor, batch_idx: int, points_yx: torch.Tensor) -> torch.Tensor:
    """在浮点坐标点集上双线性采样 feature vectors。

    输入：
        feature_map: Tensor, shape = (B, C, H, W)
        batch_idx: int
            当前样本索引。
        points_yx: Tensor, shape = (N, 2)
            全局浮点坐标，顺序为 `(y, x)`。

    输出：
        sampled_features: Tensor, shape = (N, C)
            每个点对应的特征向量。
    """
    if feature_map.dim() != 4:
        raise ValueError("feature_map must have shape (B, C, H, W)")
    if points_yx.dim() != 2 or points_yx.size(1) != 2:
        raise ValueError("points_yx must have shape (N, 2)")

    _, _, height, width = feature_map.shape
    device = feature_map.device
    points_yx = points_yx.to(device=device, dtype=torch.float32)

    if width <= 1:
        norm_x = torch.zeros_like(points_yx[:, 1])
    else:
        norm_x = 2.0 * points_yx[:, 1] / float(width - 1) - 1.0
    if height <= 1:
        norm_y = torch.zeros_like(points_yx[:, 0])
    else:
        norm_y = 2.0 * points_yx[:, 0] / float(height - 1) - 1.0

    grid = torch.stack([norm_x, norm_y], dim=1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        feature_map[int(batch_idx):int(batch_idx) + 1].to(torch.float32),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, :, 0].transpose(0, 1).contiguous()


def estimate_local_tangent_and_normal(
    boundary_points_global: torch.Tensor,
    center_point: Dict[str, float | int],
    feature_map: torch.Tensor,
    prototype_library: PrototypeLibrary,
    a: int,
    b: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """用中心点附近的一段边界点做 PCA，估计局部切线和法线。

    输出：
        tangent_vector: Tensor, shape = (2,)
            局部切线方向，坐标顺序为 `(dy, dx)`。
        normal_vector: Tensor, shape = (2,)
            局部法线方向，约定从 A 侧指向 B 侧。
    """
    if boundary_points_global.dim() != 2 or boundary_points_global.size(1) != 2:
        raise ValueError("boundary_points_global must have shape (N, 2)")
    if boundary_points_global.size(0) < 2:
        tangent = torch.tensor([0.0, 1.0], dtype=torch.float32, device=boundary_points_global.device)
        normal = torch.tensor([1.0, 0.0], dtype=torch.float32, device=boundary_points_global.device)
        return tangent, normal

    center_tensor = torch.tensor([float(center_point["y"]), float(center_point["x"] )], dtype=torch.float32, device=boundary_points_global.device)
    centered_points = boundary_points_global.to(torch.float32) - center_tensor.unsqueeze(0)
    covariance = centered_points.transpose(0, 1) @ centered_points / max(int(centered_points.size(0)), 1)
    eigen_values, eigen_vectors = torch.linalg.eigh(covariance)
    tangent = eigen_vectors[:, int(torch.argmax(eigen_values).item())]
    tangent = tangent / tangent.norm(p=2).clamp_min(1e-6)


    normal = torch.tensor([float(tangent[1].item()), -float(tangent[0].item())], dtype=torch.float32, device=tangent.device)
    normal = normal / normal.norm(p=2).clamp_min(1e-6)

    prototype_ab = _extract_prototype_from_library(prototype_library, a=a, b=b)
    prototype_ba = _extract_prototype_from_library(prototype_library, a=b, b=a)
    if prototype_ab is None or prototype_ba is None:
        return tangent, normal

    probe_distance = float(DEFAULT_STRIP_BOX_CONFIG["normal_orientation_probe_distance"])
    probe_points = torch.stack(
        [
            center_tensor - probe_distance * normal,
            center_tensor + probe_distance * normal,
        ],
        dim=0,
    )
    batch_idx = int(center_point["batch_idx"])
    probe_features = _sample_feature_vectors_at_points(feature_map=feature_map, batch_idx=batch_idx, points_yx=probe_points)
    probe_features = l2_normalize_feature(probe_features, dim=1)
    prototype_ab = l2_normalize_feature(prototype_ab.to(feature_map.device), dim=0)
    prototype_ba = l2_normalize_feature(prototype_ba.to(feature_map.device), dim=0)

    score_keep = torch.dot(probe_features[0], prototype_ab) + torch.dot(probe_features[1], prototype_ba)
    score_flip = torch.dot(probe_features[1], prototype_ab) + torch.dot(probe_features[0], prototype_ba)
    if float(score_flip.item()) > float(score_keep.item()):
        normal = -normal
    return tangent, normal


def _distance_to_box_edge_along_direction(point_yx: torch.Tensor, direction_yx: torch.Tensor, box_info: Dict[str, int]) -> float:
    """计算从点 `point_yx` 沿给定方向走到 box 边界还能走多远。"""
    distances: List[float] = []
    y0 = float(point_yx[0].item())
    x0 = float(point_yx[1].item())
    dy = float(direction_yx[0].item())
    dx = float(direction_yx[1].item())

    if abs(dy) > 1e-6:
        if dy > 0:
            distances.append((float(box_info["y_max"]) - y0) / dy)
        else:
            distances.append((float(box_info["y_min"]) - y0) / dy)
    if abs(dx) > 1e-6:
        if dx > 0:
            distances.append((float(box_info["x_max"]) - x0) / dx)
        else:
            distances.append((float(box_info["x_min"]) - x0) / dx)

    positive_distances = [distance for distance in distances if distance >= 0.0]
    if len(positive_distances) == 0:
        return 0.0
    return max(0.0, min(positive_distances))


def sample_points_along_normal(
    center_point: Dict[str, float | int],
    normal_vector: torch.Tensor,
    box: BoundaryBox,
    side_name: str,
    num_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """沿法线某一侧离散采样一串点。

    输入：
        center_point: dict
            当前局部中心点 `c`。
        normal_vector: Tensor, shape = (2,)
            局部法线方向，约定从 A 侧指向 B 侧。
        box: dict
            当前粗 box。
        side_name: str
            只能是：
            - `"a"`: 朝 A 侧，也就是沿 `-n`
            - `"b"`: 朝 B 侧，也就是沿 `+n`

    输出：
        sample_dict: dict
            {
                "points": Tensor[N, 2],
                "distances": Tensor[N],
                "direction": Tensor[2],
            }
            其中：
            - `points` 的坐标顺序是全局 `(y, x)`
            - `distances` 表示点到中心点的标量距离
            - `direction` 是当前一侧实际使用的法线方向
    """
    if side_name not in {"a", "b"}:
        raise ValueError("side_name must be either 'a' or 'b'")

    num_samples = int(DEFAULT_STRIP_BOX_CONFIG["normal_num_samples"]) if num_samples is None else int(num_samples)
    box_info = _parse_box(box)
    center_tensor = torch.tensor([float(center_point["y"]), float(center_point["x"])], dtype=torch.float32, device=normal_vector.device)
    direction = -normal_vector if side_name == "a" else normal_vector
    direction = direction.to(torch.float32)
    direction = direction / direction.norm(p=2).clamp_min(1e-6)

    max_distance = _distance_to_box_edge_along_direction(point_yx=center_tensor, direction_yx=direction, box_info=box_info)
    max_distance = max(0.0, max_distance - float(DEFAULT_STRIP_BOX_CONFIG["normal_scan_margin"]))
    if max_distance <= 1e-3:
        distances = torch.zeros(1, dtype=torch.float32, device=center_tensor.device)
        points = center_tensor.unsqueeze(0)
        return {"points": points, "distances": distances, "direction": direction}

    distances = torch.linspace(0.0, max_distance, steps=max(num_samples, 2), device=center_tensor.device)
    points = center_tensor.unsqueeze(0) + distances.unsqueeze(1) * direction.unsqueeze(0)
    return {"points": points, "distances": distances, "direction": direction}


def compute_similarity_curve_along_samples(
    feature_map: torch.Tensor,
    sample_points: torch.Tensor,
    batch_idx: int,
    prototype: torch.Tensor,
) -> torch.Tensor:
    """计算一串采样点与 prototype 的余弦相似度曲线。"""
    sampled_features = _sample_feature_vectors_at_points(feature_map=feature_map, batch_idx=batch_idx, points_yx=sample_points)
    sampled_features = l2_normalize_feature(sampled_features, dim=1)
    normalized_prototype = l2_normalize_feature(prototype.to(feature_map.device, dtype=torch.float32), dim=0)
    return torch.matmul(sampled_features, normalized_prototype)


def smooth_1d_similarity_curve(curve: torch.Tensor, kernel_size: Optional[int] = None) -> torch.Tensor:
    """对一维相似度序列做轻量平滑。

    这里使用反射 padding + 平均池化，目的是压掉单点噪声，
    同时不把整条序列抹得太平。
    """
    if curve.dim() != 1:
        raise ValueError("curve must have shape (N,)")
    kernel_size = int(DEFAULT_STRIP_BOX_CONFIG["curve_smooth_kernel"]) if kernel_size is None else int(kernel_size)
    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if curve.numel() <= 2 or kernel_size <= 1:
        return curve.to(torch.float32)

    padding = kernel_size // 2
    curve_3d = curve.to(torch.float32).view(1, 1, -1)
    padded = F.pad(curve_3d, pad=(padding, padding), mode="reflect") if curve.numel() > padding else F.pad(curve_3d, pad=(padding, padding), mode="replicate")
    smoothed = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
    return smoothed.view(-1)


def detect_similarity_changepoint(
    smoothed_curve: torch.Tensor,
    change_window: Optional[int] = None,
    min_drop: Optional[float] = None,
) -> int:
    """在一维相似度曲线上做稳定的变点检测。

    检测逻辑：
        1. 先找整条平滑曲线的主峰；
        2. 再从主峰开始往内部方向扫描；
        3. 用“左平台均值 - 右平台均值”作为突变分数；
        4. 取突变分数最大的那个位置作为变点。

    输出：
        change_index: int
            变点所在的采样索引。
    """
    if smoothed_curve.dim() != 1:
        raise ValueError("smoothed_curve must have shape (N,)")
    if smoothed_curve.numel() == 0:
        raise ValueError("smoothed_curve must not be empty")
    if smoothed_curve.numel() == 1:
        return 0

    change_window = int(DEFAULT_STRIP_BOX_CONFIG["curve_change_window"]) if change_window is None else int(change_window)
    min_drop = float(DEFAULT_STRIP_BOX_CONFIG["curve_min_drop"]) if min_drop is None else float(min_drop)
    change_window = max(1, change_window)

    peak_index = int(torch.argmax(smoothed_curve).item())
    candidate_scores: List[Tuple[float, int]] = []
    for index in range(peak_index, int(smoothed_curve.numel()) - 1):
        left_start = max(peak_index, index - change_window + 1)
        left_end = index + 1
        right_start = index + 1
        right_end = min(int(smoothed_curve.numel()), index + 1 + change_window)
        if right_start >= right_end:
            continue

        left_mean = float(smoothed_curve[left_start:left_end].mean().item())
        right_mean = float(smoothed_curve[right_start:right_end].mean().item())
        drop_score = left_mean - right_mean
        candidate_scores.append((drop_score, index))

    if len(candidate_scores) == 0:
        return int(peak_index)

    best_drop, best_index = max(candidate_scores, key=lambda item: item[0])
    if best_drop < min_drop:
        return int(peak_index)
    return int(best_index)


def construct_strip_box_from_cutoffs(
    box: BoundaryBox,
    center_point: Dict[str, float | int],
    tangent_vector: torch.Tensor,
    normal_vector: torch.Tensor,
    q_a: Dict[str, object],
    q_b: Dict[str, object],
    boundary_points_global: torch.Tensor,
) -> Dict[str, object]:
    """根据 `q_a / q_b / t` 构造最终条带框。

    构造方式：
        1. `q_a` 与 `q_b` 定义条带框在法线方向上的两条边界线；
        2. 取两者中点作为条带中心；
        3. 条带在法线方向的半宽由 `|q_b - q_a| / 2 + normal_padding` 决定；
        4. 条带在切线方向的半长由局部边界点在切线上的投影范围决定，
           并用 `strip_tangent_padding / min / max` 做稳定裁剪。
    """
    box_info = _parse_box(box)
    center_tensor = torch.tensor([float(center_point["y"]), float(center_point["x"])], dtype=torch.float32)
    q_a_tensor = torch.tensor([float(q_a["y"]), float(q_a["x"])], dtype=torch.float32)
    q_b_tensor = torch.tensor([float(q_b["y"]), float(q_b["x"])], dtype=torch.float32)

    strip_center = 0.5 * (q_a_tensor + q_b_tensor)
    half_width = 0.5 * float(torch.linalg.norm(q_b_tensor - q_a_tensor, ord=2).item())
    half_width += float(DEFAULT_STRIP_BOX_CONFIG["strip_normal_padding"])
    half_width = max(half_width, 1.5)

    relative_boundary = boundary_points_global.to(torch.float32) - strip_center.unsqueeze(0)
    tangent_projection = relative_boundary @ tangent_vector.to(torch.float32)
    if tangent_projection.numel() == 0:
        tangent_extent = 0.0
    else:
        tangent_extent = float(tangent_projection.abs().max().item())
    half_length = tangent_extent + float(DEFAULT_STRIP_BOX_CONFIG["strip_tangent_padding"])
    half_length = max(half_length, float(DEFAULT_STRIP_BOX_CONFIG["strip_min_half_length"]))
    half_length = min(half_length, float(DEFAULT_STRIP_BOX_CONFIG["strip_max_half_length"]))

    corner_1 = strip_center - half_length * tangent_vector - half_width * normal_vector
    corner_2 = strip_center + half_length * tangent_vector - half_width * normal_vector
    corner_3 = strip_center + half_length * tangent_vector + half_width * normal_vector
    corner_4 = strip_center - half_length * tangent_vector + half_width * normal_vector
    strip_polygon = torch.stack([corner_1, corner_2, corner_3, corner_4], dim=0)

    x_min = int(torch.floor(strip_polygon[:, 1].min()).item())
    y_min = int(torch.floor(strip_polygon[:, 0].min()).item())
    x_max = int(torch.ceil(strip_polygon[:, 1].max()).item())
    y_max = int(torch.ceil(strip_polygon[:, 0].max()).item())

    x_min = max(int(box_info["x_min"]), x_min)
    y_min = max(int(box_info["y_min"]), y_min)
    x_max = min(int(box_info["x_max"]), x_max)
    y_max = min(int(box_info["y_max"]), y_max)

    return {
        "batch_idx": int(box_info["batch_idx"]),
        "a": int(box.get("a", -1)),
        "b": int(box.get("b", -1)),
        "x_min": int(x_min),
        "y_min": int(y_min),
        "x_max": int(x_max),
        "y_max": int(y_max),
        "box": (int(x_min), int(y_min), int(x_max), int(y_max)),
        "strip_center_y": float(strip_center[0].item()),
        "strip_center_x": float(strip_center[1].item()),
        "strip_half_length": float(half_length),
        "strip_half_width": float(half_width),
        "tangent_vector": tangent_vector.detach().clone(),
        "normal_vector": normal_vector.detach().clone(),
        "strip_polygon": strip_polygon.detach().clone(),

        "center_point": {
            "batch_idx": int(box_info["batch_idx"]),
            "y": float(center_tensor[0].item()),
            "x": float(center_tensor[1].item()),
        },
        "q_a": dict(q_a),
        "q_b": dict(q_b),
    }


def _build_strip_coordinate_maps(strip_box: Dict[str, object]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在条带框的 axis-aligned bbox 内构造切线坐标图、法线坐标图和条带 mask。"""
    h_box, w_box = _get_box_size(_parse_box(strip_box))
    y_coords = torch.arange(h_box, dtype=torch.float32).unsqueeze(1).repeat(1, w_box) + float(strip_box["y_min"])
    x_coords = torch.arange(w_box, dtype=torch.float32).unsqueeze(0).repeat(h_box, 1) + float(strip_box["x_min"])
    relative_y = y_coords - float(strip_box["strip_center_y"])
    relative_x = x_coords - float(strip_box["strip_center_x"])
    tangent_vector = torch.as_tensor(strip_box["tangent_vector"], dtype=torch.float32)
    normal_vector = torch.as_tensor(strip_box["normal_vector"], dtype=torch.float32)

    tangent_coord = relative_y * float(tangent_vector[0].item()) + relative_x * float(tangent_vector[1].item())
    normal_coord = relative_y * float(normal_vector[0].item()) + relative_x * float(normal_vector[1].item())
    strip_mask = (tangent_coord.abs() <= float(strip_box["strip_half_length"])) & (normal_coord.abs() <= float(strip_box["strip_half_width"]))
    return tangent_coord, normal_coord, strip_mask


def compute_ordered_boundary_core_scores_in_box(
    feature_map: torch.Tensor,
    box: BoundaryBox,
    a: int,
    b: int,
    prototype_library: PrototypeLibrary,
    normalize_feature: bool = True,
    use_soft_uncertainty: bool = False,
    prob_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """兼容旧接口：在当前 box 内计算 `(a,b)` 与 `(b,a)` 两张局部响应图。

    说明：
        1. 这里保留这个函数，主要是为了让旧入口和局部调试函数别直接炸。
        2. 当前新版第三步的主流程不再把这两张图当成“核心点搜索图”，
           而是把它们当作后续条带框与 prompt 选择可以复用的 ordered prototype 响应图。
    """
    prototype_ab = _extract_prototype_from_library(prototype_library, a=a, b=b)
    prototype_ba = _extract_prototype_from_library(prototype_library, a=b, b=a)
    score_ab_box = compute_ordered_similarity_in_box(feature_map, box, prototype_ab, normalize_feature=normalize_feature, missing_proto_value=0.0)
    score_ba_box = compute_ordered_similarity_in_box(feature_map, box, prototype_ba, normalize_feature=normalize_feature, missing_proto_value=0.0)

    if use_soft_uncertainty and prob_map is not None:
        pa_box, pb_box, _ = _crop_pair_probability_maps(prob_map=prob_map, box=box, a=a, b=b)
        pair_support = torch.minimum(pa_box, pb_box)
        score_ab_box = score_ab_box * pair_support
        score_ba_box = score_ba_box * pair_support
    return score_ab_box, score_ba_box


def select_ordered_core_point_in_box(score_box: torch.Tensor, box: BoundaryBox, topk: int = 1, min_distance: float = 0.0) -> Optional[CorePoint | List[CorePoint]]:
    """兼容旧接口：从 score map 中选最高分点。

    说明：
        新版主流程已经不依赖这个函数，但保留它能避免旧实验脚本在导入时直接断掉。
    """
    if score_box.dim() != 2:
        raise ValueError("score_box must have shape (h_box, w_box)")
    box_info = _parse_box(box)
    flat_scores = score_box.reshape(-1)
    if flat_scores.numel() == 0:
        return None

    topk = max(1, int(topk))
    values, indices = torch.topk(flat_scores, k=min(topk, int(flat_scores.numel())), largest=True)
    selected_points: List[CorePoint] = []
    w_box = int(score_box.size(1))
    for value, index in zip(values.tolist(), indices.tolist()):
        y_local = int(index) // w_box
        x_local = int(index) % w_box
        point = {
            "batch_idx": int(box_info["batch_idx"]),
            "y": int(box_info["y_min"] + y_local),
            "x": int(box_info["x_min"] + x_local),
            "score": float(value),
        }
        if min_distance > 0 and len(selected_points) > 0:
            too_close = False
            for selected in selected_points:
                dy = float(point["y"]) - float(selected["y"])
                dx = float(point["x"]) - float(selected["x"])
                if math.sqrt(dy * dy + dx * dx) < float(min_distance):
                    too_close = True
                    break
            if too_close:
                continue
        selected_points.append(point)
    if len(selected_points) == 0:
        return None
    if topk == 1:
        return selected_points[0]
    return selected_points


def generate_ordered_core_points_in_box(
    feature_map: torch.Tensor,
    box: BoundaryBox,
    a: int,
    b: int,
    prototype_library: PrototypeLibrary,
    normalize_feature: bool = True,
    use_soft_uncertainty: bool = False,
    prob_map: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """新版第三步主函数：从粗边界 box 构造局部条带框。

    输入：
        feature_map: Tensor, shape = (B, C, H, W)
            目标域特征图。
        box: dict
            当前粗定位 box。
        a: int
            当前 pair 的第一类 id。
        b: int
            当前 pair 的第二类 id。
        prototype_library:
            ordered boundary prototype library。
        normalize_feature: bool
            保留兼容参数，当前内部相似度默认都基于归一化向量。
        use_soft_uncertainty: bool
            保留兼容参数。若为 True 且提供 `prob_map`，局部响应图会乘一点 pair support。
        prob_map: Optional[Tensor], shape = (B, K, H, W)
            教师模型概率图。新版第三步依赖它来从粗边界附近估中心点和提局部边界。

    输出：
        result_dict: dict
            主要字段包括：
            {
                "pair": (a, b),
                "box": {...},
                "core_ab": {...},
                "core_ba": {...},
                "center_point": {...},
                "tangent_vector": Tensor[2],
                "normal_vector": Tensor[2],
                "q_a": {...},
                "q_b": {...},
                "boundary_points_global": Tensor[N, 2],
                "boundary_response_box": Tensor[h_box, w_box],
                "normal_samples_a": {...},
                "normal_samples_b": {...},
                "similarity_curve_a": Tensor[N],
                "similarity_curve_b": Tensor[N],
                "smooth_similarity_curve_a": Tensor[N],
                "smooth_similarity_curve_b": Tensor[N],
                "changepoint_index_a": int,
                "changepoint_index_b": int,
                "response_a_box": Tensor[h_strip, w_strip],
                "response_b_box": Tensor[h_strip, w_strip],
                "tangent_coord_box": Tensor[h_strip, w_strip],
                "normal_coord_box": Tensor[h_strip, w_strip],
                "strip_mask_box": Tensor[h_strip, w_strip],
            }
            其中：
            - `box` 保存最终条带框的轴对齐 bbox 与几何元信息
            - `core_ab` 与 `core_ba` 是兼容旧接口的字段，分别别名到 `q_a` 与 `q_b`
    """
    if prob_map is None:
        raise ValueError("prob_map must be provided in the new step-three pipeline")
    if feature_map.dim() != 4:
        raise ValueError("feature_map must have shape (B, C, H, W)")

    prototype_ab = _extract_prototype_from_library(prototype_library, a=a, b=b)
    prototype_ba = _extract_prototype_from_library(prototype_library, a=b, b=a)
    if prototype_ab is None or prototype_ba is None:
        raise RuntimeError(f"Missing ordered prototypes for pair ({a}, {b})")

    coarse_boundary_coords = box.get("boundary_coords") if isinstance(box, dict) else None
    center_point = extract_local_center_point_from_coarse_boundary(
        prob_map=prob_map,
        box=box,
        a=int(a),
        b=int(b),
        coarse_boundary_coords=coarse_boundary_coords if isinstance(coarse_boundary_coords, torch.Tensor) else None,
    )
    boundary_info = extract_local_boundary_points_near_center(
        prob_map=prob_map,
        box=box,
        a=int(a),
        b=int(b),
        center_point=center_point,
        coarse_boundary_coords=coarse_boundary_coords if isinstance(coarse_boundary_coords, torch.Tensor) else None,
    )
    tangent_vector, normal_vector = estimate_local_tangent_and_normal(
        boundary_points_global=boundary_info["boundary_points_global"],
        center_point=center_point,
        feature_map=feature_map,
        prototype_library=prototype_library,
        a=int(a),
        b=int(b),
    )

    samples_a = sample_points_along_normal(center_point=center_point, normal_vector=normal_vector, box=box, side_name="a")
    samples_b = sample_points_along_normal(center_point=center_point, normal_vector=normal_vector, box=box, side_name="b")
    similarity_curve_a = compute_similarity_curve_along_samples(
        feature_map=feature_map,
        sample_points=samples_a["points"],
        batch_idx=int(center_point["batch_idx"]),
        prototype=prototype_ab,
    )
    similarity_curve_b = compute_similarity_curve_along_samples(
        feature_map=feature_map,
        sample_points=samples_b["points"],
        batch_idx=int(center_point["batch_idx"]),
        prototype=prototype_ba,
    )
    smooth_curve_a = smooth_1d_similarity_curve(similarity_curve_a)
    smooth_curve_b = smooth_1d_similarity_curve(similarity_curve_b)
    changepoint_index_a = detect_similarity_changepoint(smooth_curve_a)
    changepoint_index_b = detect_similarity_changepoint(smooth_curve_b)

    q_a_tensor = samples_a["points"][int(changepoint_index_a)]
    q_b_tensor = samples_b["points"][int(changepoint_index_b)]
    q_a = {
        "batch_idx": int(center_point["batch_idx"]),
        "y": float(q_a_tensor[0].item()),
        "x": float(q_a_tensor[1].item()),
        "score": float(smooth_curve_a[int(changepoint_index_a)].item()),
        "ordered_boundary_key": canonicalize_ordered_boundary_key(int(a), int(b)),
    }
    q_b = {
        "batch_idx": int(center_point["batch_idx"]),
        "y": float(q_b_tensor[0].item()),
        "x": float(q_b_tensor[1].item()),
        "score": float(smooth_curve_b[int(changepoint_index_b)].item()),
        "ordered_boundary_key": canonicalize_ordered_boundary_key(int(b), int(a)),
    }

    strip_box = construct_strip_box_from_cutoffs(
        box=box,
        center_point=center_point,
        tangent_vector=tangent_vector,
        normal_vector=normal_vector,
        q_a=q_a,
        q_b=q_b,
        boundary_points_global=boundary_info["boundary_points_global"],
    )

    response_a_box = compute_ordered_similarity_in_box(
        feature_map=feature_map,
        box=strip_box,
        prototype=prototype_ab,
        normalize_feature=normalize_feature,
        missing_proto_value=0.0,
    )
    response_b_box = compute_ordered_similarity_in_box(
        feature_map=feature_map,
        box=strip_box,
        prototype=prototype_ba,
        normalize_feature=normalize_feature,
        missing_proto_value=0.0,
    )
    if use_soft_uncertainty:
        pa_strip, pb_strip, _ = _crop_pair_probability_maps(prob_map=prob_map, box=strip_box, a=int(a), b=int(b))
        pair_support_strip = torch.minimum(pa_strip, pb_strip)
        response_a_box = response_a_box * pair_support_strip
        response_b_box = response_b_box * pair_support_strip

    tangent_coord_box, normal_coord_box, strip_mask_box = _build_strip_coordinate_maps(strip_box)
    strip_box["response_a_box"] = response_a_box
    strip_box["response_b_box"] = response_b_box
    strip_box["tangent_coord_box"] = tangent_coord_box
    strip_box["normal_coord_box"] = normal_coord_box
    strip_box["strip_mask_box"] = strip_mask_box


    strip_box["boundary_points_global"] = boundary_info["boundary_points_global"]
    strip_box["normal_samples_a"] = samples_a
    strip_box["normal_samples_b"] = samples_b
    strip_box["similarity_curve_a"] = similarity_curve_a
    strip_box["similarity_curve_b"] = similarity_curve_b
    strip_box["smooth_similarity_curve_a"] = smooth_curve_a
    strip_box["smooth_similarity_curve_b"] = smooth_curve_b
    strip_box["changepoint_index_a"] = int(changepoint_index_a)
    strip_box["changepoint_index_b"] = int(changepoint_index_b)

    return {
        "pair": canonicalize_ordered_boundary_key(int(a), int(b)),
        "box": strip_box,
        "core_ab": dict(q_a),
        "core_ba": dict(q_b),
        "center_point": center_point,
        "tangent_vector": tangent_vector.detach().clone(),
        "normal_vector": normal_vector.detach().clone(),
        "q_a": q_a,
        "q_b": q_b,
        "boundary_points_global": boundary_info["boundary_points_global"],
        "boundary_points_local": boundary_info["boundary_points_local"],
        "boundary_scores": boundary_info["boundary_scores"],
        "boundary_response_box": boundary_info["boundary_response_box"],
        "normal_samples_a": samples_a,
        "normal_samples_b": samples_b,
        "similarity_curve_a": similarity_curve_a,
        "similarity_curve_b": similarity_curve_b,
        "smooth_similarity_curve_a": smooth_curve_a,
        "smooth_similarity_curve_b": smooth_curve_b,
        "changepoint_index_a": int(changepoint_index_a),
        "changepoint_index_b": int(changepoint_index_b),
        "response_a_box": response_a_box,
        "response_b_box": response_b_box,
        "tangent_coord_box": tangent_coord_box,
        "normal_coord_box": normal_coord_box,
        "strip_mask_box": strip_mask_box,
    }


def generate_ordered_core_points_for_boxes(
    feature_map: torch.Tensor,
    box_list: List[BoundaryBox],
    prototype_library: PrototypeLibrary,
    prob_map: Optional[torch.Tensor] = None,
    use_soft_uncertainty: bool = False,
) -> List[Dict[str, object]]:
    """对多个粗 box 批量构造条带框。

    输入：
        feature_map: Tensor, shape = (B, C, H, W)
        box_list: list
            每个元素至少包含：
            - `batch_idx`
            - `a`
            - `b`
            - `box` 或 `x_min/y_min/x_max/y_max`
        prototype_library:
            ordered prototype library。
        prob_map: Optional[Tensor], shape = (B, K, H, W)
            新版第三步需要它来估中心点和局部粗边界。

    输出：
        result_list: list
            每个粗 box 对应一个条带框构造结果。
    """
    if prob_map is None:
        raise ValueError("prob_map must be provided in the new step-three pipeline")

    result_list: List[Dict[str, object]] = []
    for box_record in box_list:
        if not isinstance(box_record, dict):
            raise TypeError("each item in box_list must be a dictionary")
        if "a" not in box_record or "b" not in box_record:
            raise KeyError("each box must contain 'a' and 'b'")

        result = generate_ordered_core_points_in_box(
            feature_map=feature_map,
            box=box_record,
            a=int(box_record["a"]),
            b=int(box_record["b"]),
            prototype_library=prototype_library,
            normalize_feature=True,
            use_soft_uncertainty=use_soft_uncertainty,
            prob_map=prob_map,
        )
        result_list.append(result)
    return result_list


def visualize_ordered_core_points_in_box(
    image_or_mask_2d: torch.Tensor,
    result: Dict[str, object],
    save_path: str | Path,
) -> None:
    """可视化新版第三步的条带框构造过程。

    这张调试图会画出：
        1. 粗边界局部点
        2. 中心点 `c`
        3. 切线 `t`
        4. 法线 `n`
        5. 法线两侧采样点
        6. 两个变点 `q_a / q_b`
        7. 最终条带框
        8. A / B 两条一维相似度曲线与变点位置
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    if not isinstance(image_or_mask_2d, torch.Tensor):
        image_or_mask_2d = torch.as_tensor(image_or_mask_2d)
    if image_or_mask_2d.dim() != 2:
        raise ValueError("image_or_mask_2d must have shape (H, W)")

    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    box = result["box"]
    box_info = _parse_box(box)
    image_np = image_or_mask_2d.detach().cpu().numpy()
    use_label_cmap = not torch.is_floating_point(image_or_mask_2d)
    cmap = "nipy_spectral" if use_label_cmap else "gray"

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2))
    axis_img, axis_curve = axes
    axis_img.imshow(image_np, cmap=cmap, interpolation="nearest")

    rough_rect = Rectangle(
        (box_info["x_min"], box_info["y_min"]),
        box_info["x_max"] - box_info["x_min"] + 1,
        box_info["y_max"] - box_info["y_min"] + 1,
        fill=False,
        edgecolor="yellow",
        linewidth=1.6,
        linestyle="--",
        alpha=0.8,
    )
    axis_img.add_patch(rough_rect)

    boundary_points = result["boundary_points_global"]
    axis_img.scatter(
        boundary_points[:, 1].detach().cpu().numpy(),
        boundary_points[:, 0].detach().cpu().numpy(),
        s=10,
        c="white",
        linewidths=0.0,
        alpha=0.65,
        label="local coarse boundary",
    )

    center_point = result["center_point"]
    tangent_vector = torch.as_tensor(result["tangent_vector"], dtype=torch.float32)
    normal_vector = torch.as_tensor(result["normal_vector"], dtype=torch.float32)
    center_y = float(center_point["y"])
    center_x = float(center_point["x"])
    axis_img.scatter([center_x], [center_y], s=85, c="magenta", edgecolors="black", linewidths=0.7, label="center c")

    tangent_scale = float(box["strip_half_length"])
    normal_scale = float(box["strip_half_width"])
    axis_img.plot(
        [center_x - tangent_scale * float(tangent_vector[1]), center_x + tangent_scale * float(tangent_vector[1])],
        [center_y - tangent_scale * float(tangent_vector[0]), center_y + tangent_scale * float(tangent_vector[0])],
        color="magenta",
        linestyle=":",
        linewidth=1.5,
        alpha=0.9,
        label="tangent t",
    )
    axis_img.arrow(
        center_x,
        center_y,
        normal_scale * float(normal_vector[1]),
        normal_scale * float(normal_vector[0]),
        color="magenta",
        width=0.16,
        head_width=1.0,
        length_includes_head=True,
        alpha=0.9,
    )
    axis_img.arrow(
        center_x,
        center_y,
        -normal_scale * float(normal_vector[1]),
        -normal_scale * float(normal_vector[0]),
        color="magenta",
        width=0.10,
        head_width=0.9,
        length_includes_head=True,
        alpha=0.5,
    )

    for sample_key, sample_color, sample_label in (("normal_samples_a", "orange", "A normal samples"), ("normal_samples_b", "lime", "B normal samples")):
        sample_points = result[sample_key]["points"]
        axis_img.scatter(
            sample_points[:, 1].detach().cpu().numpy(),
            sample_points[:, 0].detach().cpu().numpy(),
            s=12,
            c=sample_color,
            alpha=0.55,
            linewidths=0.0,
            label=sample_label,
        )

    q_a = result["q_a"]
    q_b = result["q_b"]
    axis_img.scatter([float(q_a["x"])], [float(q_a["y"])], s=120, c="red", marker="*", edgecolors="white", linewidths=0.8, label="q_a")
    axis_img.scatter([float(q_b["x"])], [float(q_b["y"])], s=120, c="cyan", marker="*", edgecolors="black", linewidths=0.8, label="q_b")

    strip_polygon = torch.as_tensor(box["strip_polygon"], dtype=torch.float32)
    polygon = Polygon(
        strip_polygon[:, [1, 0]].detach().cpu().numpy(),
        closed=True,
        fill=False,
        edgecolor="yellow",
        linewidth=2.0,
        linestyle="-",
        alpha=0.95,
    )
    axis_img.add_patch(polygon)
    axis_img.set_title(f"pair={tuple(result['pair'])}, strip box construction")
    axis_img.axis("off")
    axis_img.legend(loc="upper right", framealpha=0.9, fontsize=8)

    distances_a = result["normal_samples_a"]["distances"].detach().cpu().numpy()
    distances_b = result["normal_samples_b"]["distances"].detach().cpu().numpy()
    curve_a = result["similarity_curve_a"].detach().cpu().numpy()
    curve_b = result["similarity_curve_b"].detach().cpu().numpy()
    smooth_a = result["smooth_similarity_curve_a"].detach().cpu().numpy()
    smooth_b = result["smooth_similarity_curve_b"].detach().cpu().numpy()
    idx_a = int(result["changepoint_index_a"])
    idx_b = int(result["changepoint_index_b"])

    axis_curve.plot(distances_a, curve_a, color="orange", alpha=0.35, linewidth=1.2, label="A raw")
    axis_curve.plot(distances_a, smooth_a, color="orange", alpha=0.95, linewidth=2.0, label="A smooth")
    axis_curve.axvline(distances_a[idx_a], color="red", linestyle="--", linewidth=1.4, label="A change")
    axis_curve.plot(distances_b, curve_b, color="limegreen", alpha=0.35, linewidth=1.2, label="B raw")
    axis_curve.plot(distances_b, smooth_b, color="limegreen", alpha=0.95, linewidth=2.0, label="B smooth")
    axis_curve.axvline(distances_b[idx_b], color="cyan", linestyle="--", linewidth=1.4, label="B change")
    axis_curve.set_title("prototype similarity along local normal")
    axis_curve.set_xlabel("distance from center c")
    axis_curve.set_ylabel("cosine similarity")
    axis_curve.grid(alpha=0.25)
    axis_curve.legend(loc="best", framealpha=0.9, fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_all_ordered_boundary_scores(*args: object, **kwargs: object) -> None:
    """旧版全图接口已废弃。"""
    raise NotImplementedError(
        "The old full-image ordered-boundary score interface has been removed. "
        "Use `generate_ordered_core_points_in_box` or `generate_ordered_core_points_for_boxes` instead."
    )


def extract_ordered_boundary_prompt_seeds(*args: object, **kwargs: object) -> None:
    """旧版 seed 提取接口已废弃。"""
    raise NotImplementedError(
        "The old prompt-seed interface has been removed. "
        "Use the new strip-box third step plus the segmented prompt generation in part four."
    )


def _build_demo_inputs() -> Tuple[torch.Tensor, torch.Tensor, Dict[BoundaryKey, torch.Tensor], Dict[str, object]]:
    """构造一个最小可运行 demo 所需的 feature/prob/prototype/box。"""
    height, width = 72, 72
    y_coords = torch.arange(height, dtype=torch.float32).unsqueeze(1).repeat(1, width)
    x_coords = torch.arange(width, dtype=torch.float32).unsqueeze(0).repeat(height, 1)


    center_x = 36.0
    distance_x = x_coords - center_x
    prob_a = torch.sigmoid(-distance_x / 2.6)
    prob_b = 1.0 - prob_a
    prob_bg = torch.zeros_like(prob_a)
    prob_map = torch.stack([prob_bg, prob_a, prob_b], dim=0).unsqueeze(0)
    prob_map = prob_map / prob_map.sum(dim=1, keepdim=True).clamp_min(1e-6)

    feature_0 = (x_coords - center_x) / 12.0
    feature_1 = (y_coords - 36.0) / 16.0
    feature_2 = torch.exp(-(distance_x ** 2) / 40.0)
    feature_3 = torch.exp(-((y_coords - 36.0) ** 2) / 110.0)
    feature_map = torch.stack([feature_0, feature_1, feature_2, feature_3], dim=0).unsqueeze(0).to(torch.float32)

    prototype_a = torch.tensor([-0.8, 0.0, 0.9, 0.2], dtype=torch.float32)
    prototype_b = torch.tensor([0.8, 0.0, 0.9, 0.2], dtype=torch.float32)
    prototype_library = {
        (1, 2): prototype_a,
        (2, 1): prototype_b,
    }

    box = {
        "batch_idx": 0,
        "a": 1,
        "b": 2,
        "box": (18, 16, 54, 56),
    }
    return feature_map, prob_map, prototype_library, box


def run_demo(save_dir: str | Path) -> Dict[str, object]:
    """运行第三部分最小 demo，并保存一张调试图。"""
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    feature_map, prob_map, prototype_library, box = _build_demo_inputs()
    result = generate_ordered_core_points_in_box(
        feature_map=feature_map,
        box=box,
        a=1,
        b=2,
        prototype_library=prototype_library,
        prob_map=prob_map,
    )
    visualize_ordered_core_points_in_box(
        image_or_mask_2d=torch.argmax(prob_map[0], dim=0),
        result=result,
        save_path=save_dir / "ordered_boundary_strip_box_demo.png",
    )
    print(f"center_point={result['center_point']}")
    print(f"q_a={result['q_a']}")
    print(f"q_b={result['q_b']}")
    print(f"strip_box={result['box']['box']}")
    return result


def main() -> None:
    """模块直接运行时，执行最小 demo。"""
    run_demo(Path(__file__).resolve().parent / "outputs")


if __name__ == "__main__":
    main()
