"""第四部分：基于两个有序核心边界点生成分层、多点、边界友好的 SAM point prompts。

当前文件只修改“从中心点到最终 prompt”的这一步，不推翻前面三部分已有流程。

前面三部分仍然负责：
1. 用伪标签识别局部边界类型；
2. 在局部 box 内找到有序核心边界点；
3. 给出当前 pair 的几何粗定位。

当前第四部分负责：
1. 用 `core_ab / core_ba` 推导局部中心点与局部法线；
2. 沿法线两侧搜索两个起始锚点；
3. 围绕锚点按分层半径生成 prompt；
4. 用“连线边界一致性”抑制跨边界的坏点；
5. 输出更沿边界分散、而不是向某个高置信度小块塌缩的 prompts。

注意：
1. 当前模块不实现 box prompt。
2. 当前模块不直接调用 SAM。
3. 当前模块不改变前面三部分的输入输出接口。
4. 当前模块优先保证流程清楚、可解释、可视化友好。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


PointDict = Dict[str, int | float | str | Tuple[int, int]]
BoxDict = Dict[str, int | float | Tuple[int, int, int, int]]


# 这里把第四部分内部用到的超参数集中放在一起，别散落在多处。
DEFAULT_PROMPT_GENERATION_CONFIG: Dict[str, float | int] = {
    "boundary_fit_radius": 14.0,  # 拟合法线时，中心点附近最多看多大的局部半径。
    "boundary_fit_topk": 96,  # 用于切线/法线拟合的伪边界点最多保留多少个。
    "boundary_response_temperature": 0.20,  # `|p_a-p_b|` 转边界响应时的温度项，越小越强调“接近平手”。
    "boundary_min_pair_support": 0.05,  # 伪边界点至少要让两类都有一点存在感，太低就没必要当边界看。
    "normal_side_margin": 0.25,  # 判断点属于法线哪一侧时留一点小 margin，别把刚好压边界的点算进去。
    "anchor_search_extra_distance": 6.0,  # 锚点搜索区间在 `offset_distance` 基础上额外向外延长多少像素。
    "anchor_num_samples": 25,  # 每一侧一维搜索区间采样多少个候选锚点。
    "anchor_distance_sigma": 2.0,  # 锚点距离惩罚的高斯尺度。
    "anchor_boundary_penalty_gamma": 2.0,  # 锚点边界惩罚指数，越大越不愿意把锚点压在边界上。
    "desired_boundary_distance": 3.0,  # prompt 更偏好离伪边界多少像素，不要太贴边，也不要跑太深。
    "boundary_distance_sigma": 1.5,  # 对“离边界多远才舒服”的高斯尺度。
    "boundary_distance_min": 1.0,  # prompt 至少离边界这么远，避免直接压在边界上。
    "boundary_distance_max": 7.0,  # prompt 最多离边界这么远，避免钻到器官深处或背景深处。
    "consistency_lambda": 2.0,  # 连线边界一致性指数，越大越惩罚中途穿边界的候选点。
}


def _parse_box(box: BoxDict) -> Dict[str, int]:
    """把输入 box 解析成统一格式。

    标准 box 格式建议为：
        {
            "batch_idx": int,
            "a": int,
            "b": int,
            "x_min": int,
            "y_min": int,
            "x_max": int,
            "y_max": int,
        }

    兼容格式：
        1. 使用 `x1, y1, x2, y2`
        2. 使用 `box=(x1, y1, x2, y2)`

    输出：
        parsed_box: dict
            统一后的闭区间 box：
            {
                "batch_idx": int,
                "a": int,
                "b": int,
                "x_min": int,
                "y_min": int,
                "x_max": int,
                "y_max": int,
            }
    """
    if not isinstance(box, dict):
        raise TypeError("box must be a dictionary")

    batch_idx = int(box.get("batch_idx", 0))
    a = int(box.get("a", -1))
    b = int(box.get("b", -1))

    if "box" in box:
        raw_box = box["box"]
        if not isinstance(raw_box, (tuple, list)) or len(raw_box) != 4:
            raise ValueError("box['box'] must be a tuple/list with four elements: (x_min, y_min, x_max, y_max)")
        x_min, y_min, x_max, y_max = [int(v) for v in raw_box]
    else:
        x_min = int(box["x_min"]) if "x_min" in box else int(box["x1"])
        y_min = int(box["y_min"]) if "y_min" in box else int(box["y1"])
        x_max = int(box["x_max"]) if "x_max" in box else int(box["x2"])
        y_max = int(box["y_max"]) if "y_max" in box else int(box["y2"])

    if x_max < x_min or y_max < y_min:
        raise ValueError("box must satisfy x_max >= x_min and y_max >= y_min")

    return {
        "batch_idx": batch_idx,
        "a": a,
        "b": b,
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
    }


def _parse_point(point: PointDict) -> Dict[str, float | int | str | Tuple[int, int]]:
    """把点字典解析成统一格式。

    标准点格式建议为：
        {
            "batch_idx": int,
            "y": int or float,
            "x": int or float,
            "score": float,                     # 可选
            "ordered_boundary_key": (A, B),    # 可选
        }

    输出：
        parsed_point: dict
            至少包含：
            - `batch_idx`
            - `y`
            - `x`
    """
    if not isinstance(point, dict):
        raise TypeError("point must be a dictionary")
    if "batch_idx" not in point or "y" not in point or "x" not in point:
        raise KeyError("point must contain 'batch_idx', 'y', and 'x'")

    parsed_point: Dict[str, float | int | str | Tuple[int, int]] = dict(point)
    parsed_point["batch_idx"] = int(point["batch_idx"])
    parsed_point["y"] = float(point["y"])
    parsed_point["x"] = float(point["x"])
    if "score" in point:
        parsed_point["score"] = float(point["score"])
    return parsed_point


def point_to_tensor(point: PointDict) -> torch.Tensor:
    """把点字典转换成二维坐标张量。

    输入：
        point: dict
            至少包含：
            - `y`
            - `x`

    输出：
        coord_tensor: Tensor, shape = (2,)
            坐标顺序固定为 `(y, x)`。
    """
    parsed_point = _parse_point(point)
    return torch.tensor([float(parsed_point["y"]), float(parsed_point["x"])], dtype=torch.float32)


def normalize_vector(v: torch.Tensor, eps: float = 1e-6) -> Optional[torch.Tensor]:
    """对二维向量做单位化。

    输入：
        v: Tensor, shape = (2,)
            二维向量，顺序为 `(dy, dx)`。
        eps: float
            数值稳定项。

    输出：
        normalized_v: Optional[Tensor], shape = (2,)
            若向量长度足够大，则返回单位向量。
            若长度过小，则返回 `None`。
    """
    if not isinstance(v, torch.Tensor):
        raise TypeError("v must be a torch.Tensor")
    if v.numel() != 2:
        raise ValueError("v must have shape (2,)")

    v = v.to(torch.float32).reshape(2)
    norm = torch.linalg.norm(v, ord=2)
    if float(norm.item()) < float(eps):
        return None
    return v / (norm + float(eps))


def clip_point_to_box(point: PointDict | torch.Tensor, box: BoxDict) -> Dict[str, float | int]:
    """把点裁剪到当前 box 内。

    输入：
        point:
            1. dict：至少包含 `batch_idx, y, x`
            2. Tensor, shape = (2,)：顺序为 `(y, x)`
        box: dict
            当前局部框，闭区间坐标。

    输出：
        clipped_point: dict
            {
                "batch_idx": int,
                "y": float,
                "x": float,
            }
    """
    parsed_box = _parse_box(box)

    if isinstance(point, torch.Tensor):
        if point.numel() != 2:
            raise ValueError("point tensor must have shape (2,)")
        batch_idx = int(parsed_box["batch_idx"])
        y = float(point.reshape(2)[0].item())
        x = float(point.reshape(2)[1].item())
    else:
        parsed_point = _parse_point(point)
        batch_idx = int(parsed_point["batch_idx"])
        y = float(parsed_point["y"])
        x = float(parsed_point["x"])

    y = min(max(y, float(parsed_box["y_min"])), float(parsed_box["y_max"]))
    x = min(max(x, float(parsed_box["x_min"])), float(parsed_box["x_max"]))
    return {
        "batch_idx": batch_idx,
        "y": y,
        "x": x,
    }


def clip_box_to_image(box: BoxDict, height: int, width: int) -> Dict[str, int]:
    """把 box 裁剪到图像范围内。

    输入：
        box: dict
            局部框，闭区间坐标。
        height: int
            图像高度 H。
        width: int
            图像宽度 W。

    输出：
        clipped_box: dict
            与 `_parse_box` 的输出格式一致，但坐标被裁剪到了图像范围内。
    """
    parsed_box = _parse_box(box)
    x_min = min(max(int(parsed_box["x_min"]), 0), int(width) - 1)
    y_min = min(max(int(parsed_box["y_min"]), 0), int(height) - 1)
    x_max = min(max(int(parsed_box["x_max"]), 0), int(width) - 1)
    y_max = min(max(int(parsed_box["y_max"]), 0), int(height) - 1)
    if x_max < x_min:
        x_max = x_min
    if y_max < y_min:
        y_max = y_min

    parsed_box["x_min"] = x_min
    parsed_box["y_min"] = y_min
    parsed_box["x_max"] = x_max
    parsed_box["y_max"] = y_max
    return parsed_box


def compute_local_boundary_direction(
    core_ab: PointDict,
    core_ba: PointDict,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """根据两个有序核心点计算粗跨边界方向。

    输入：
        core_ab: dict
            第三部分输出的 `z_{A,B}`。
        core_ba: dict
            第三部分输出的 `z_{B,A}`。
        eps: float
            数值稳定项。

    输出：
        coarse_direction: Optional[Tensor], shape = (2,)
            由 `core_ba - core_ab` 直接得到的粗方向。

    说明：
        1. 这只是粗方向，不是最终用来生成 prompt 的局部法线。
        2. 真正的局部法线会优先由“中心点附近的伪边界做 PCA”得到。
        3. 当前函数主要作为拟合法线失败时的后备方向。
    """
    point_ab = point_to_tensor(core_ab)
    point_ba = point_to_tensor(core_ba)
    delta = point_ba - point_ab
    return normalize_vector(delta, eps=eps)


def compute_center_point_from_ordered_cores(
    core_ab: PointDict,
    core_ba: PointDict,
) -> Dict[str, float | int]:
    """由两个有序核心点计算当前边界对的局部中心点。

    输入：
        core_ab: dict
            `z_{A,B}`。
        core_ba: dict
            `z_{B,A}`。

    输出：
        center_point: dict
            {
                "batch_idx": int,
                "y": float,
                "x": float,
            }

    说明：
        1. 这里直接取两个有序核心点的中点。
        2. 这个中心点代表“当前局部边界段的大致中心”，后面会围绕它拟合法线。
    """
    parsed_core_ab = _parse_point(core_ab)
    parsed_core_ba = _parse_point(core_ba)
    if int(parsed_core_ab["batch_idx"]) != int(parsed_core_ba["batch_idx"]):
        raise ValueError("core_ab and core_ba must belong to the same batch index")

    center_y = 0.5 * (float(parsed_core_ab["y"]) + float(parsed_core_ba["y"]))
    center_x = 0.5 * (float(parsed_core_ab["x"]) + float(parsed_core_ba["x"]))
    return {
        "batch_idx": int(parsed_core_ab["batch_idx"]),
        "y": center_y,
        "x": center_x,
    }


def _crop_pair_probability_maps(
    prob_map: torch.Tensor,
    box: BoxDict,
    a: int,
    b: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """裁出当前 pair 在 box 内的两张类别概率图。"""
    if prob_map.dim() != 4:
        raise ValueError("prob_map must have shape (B, K, H, W)")

    batch_size, num_classes, height, width = prob_map.shape
    if not (0 <= int(a) < num_classes and 0 <= int(b) < num_classes):
        raise IndexError("a and b must be valid class indices for prob_map")

    parsed_box = clip_box_to_image(box, height=height, width=width)
    batch_idx = int(parsed_box["batch_idx"])
    if not (0 <= batch_idx < batch_size):
        raise IndexError("box batch_idx is out of range")

    pa_box = prob_map[
        batch_idx,
        int(a),
        parsed_box["y_min"]:parsed_box["y_max"] + 1,
        parsed_box["x_min"]:parsed_box["x_max"] + 1,
    ].to(torch.float32)
    pb_box = prob_map[
        batch_idx,
        int(b),
        parsed_box["y_min"]:parsed_box["y_max"] + 1,
        parsed_box["x_min"]:parsed_box["x_max"] + 1,
    ].to(torch.float32)
    return pa_box, pb_box


def extract_local_boundary_points(
    prob_map: torch.Tensor,
    box: BoxDict,
    a: int,
    b: int,
    center_point: PointDict,
    fit_radius: Optional[float] = None,
    topk_points: Optional[int] = None,
) -> Dict[str, object]:
    """在当前 box 内提取一小段用于拟合切线/法线的局部伪边界点。

    输入：
        prob_map: Tensor, shape = (B, K, H, W)
            教师模型概率图。
        box: dict
            当前局部 box。
        a: int
            当前 pair 中的 A 类别。
        b: int
            当前 pair 中的对侧类别，可以是背景 0，也可以是器官 B。
        center_point: dict
            当前局部边界中心点，通常取 `core_ab / core_ba` 的中点。
        fit_radius: Optional[float]
            用于拟合法线的局部半径。
            若为 `None`，则使用默认超参数 `boundary_fit_radius`。
        topk_points: Optional[int]
            最多保留多少个局部伪边界点。
            若为 `None`，则使用默认超参数 `boundary_fit_topk`。

    输出：
        boundary_info: dict
            {
                "boundary_response_box": Tensor[h_box, w_box],
                "boundary_points_global": Tensor[N, 2],   # 全局坐标 `(y, x)`
                "boundary_points_local": Tensor[N, 2],    # box 内局部坐标 `(y_local, x_local)`
                "boundary_mask_box": Tensor[h_box, w_box],# bool
            }

    说明：
        1. 这里不改前面伪标签边界类型识别，只在已知 pair=(A,B) 的前提下抽一小段局部伪边界。
        2. 伪边界响应用下面这个 deterministic 形式：
           - `pair_support = min(p_a, p_b)`
           - `gap = |p_a - p_b|`
           - `boundary_response = pair_support * exp(-gap / temperature)`
        3. 这一步的目的不是追求全局精确边界，而是给局部切线/法线估计提供一小段几何支撑。
    """
    fit_radius = float(DEFAULT_PROMPT_GENERATION_CONFIG["boundary_fit_radius"]) if fit_radius is None else float(fit_radius)
    topk_points = int(DEFAULT_PROMPT_GENERATION_CONFIG["boundary_fit_topk"]) if topk_points is None else int(topk_points)

    pa_box, pb_box = _crop_pair_probability_maps(prob_map=prob_map, box=box, a=a, b=b)
    parsed_box = _parse_box(box)
    parsed_center = _parse_point(center_point)

    pair_support = torch.minimum(pa_box, pb_box)  # 只有两类都“有点像”时，这个位置才像真正的 pair 边界。
    probability_gap = torch.abs(pa_box - pb_box)  # 两类概率差越小，越像边界过渡带。
    temperature = float(DEFAULT_PROMPT_GENERATION_CONFIG["boundary_response_temperature"])
    boundary_response_box = pair_support * torch.exp(-probability_gap / max(temperature, 1e-6))

    height_box, width_box = boundary_response_box.shape
    y_local = torch.arange(height_box, dtype=torch.float32, device=prob_map.device)
    x_local = torch.arange(width_box, dtype=torch.float32, device=prob_map.device)
    grid_y_local, grid_x_local = torch.meshgrid(y_local, x_local, indexing="ij")
    grid_y_global = grid_y_local + float(parsed_box["y_min"])
    grid_x_global = grid_x_local + float(parsed_box["x_min"])

    center_distance = torch.sqrt(
        (grid_y_global - float(parsed_center["y"])) ** 2 +
        (grid_x_global - float(parsed_center["x"])) ** 2
    )  # 这里只拿中心点附近那一小段伪边界做拟合，别把整块 box 都混进去。
    local_radius_mask = center_distance <= float(fit_radius)
    pair_support_mask = pair_support >= float(DEFAULT_PROMPT_GENERATION_CONFIG["boundary_min_pair_support"])

    candidate_response = torch.where(
        local_radius_mask & pair_support_mask,
        boundary_response_box,
        torch.full_like(boundary_response_box, fill_value=-1.0),
    )  # 非局部区域直接设成负数，后面 top-k 时就自然不会被选上。

    flat_candidate_response = candidate_response.reshape(-1)
    valid_mask = flat_candidate_response > 0.0
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    if valid_indices.numel() == 0:
        # 局部伪边界完全提不出来时，至少返回空结构，别让后续 silently 出错。
        empty_coords = torch.zeros((0, 2), dtype=torch.float32, device=prob_map.device)
        return {
            "boundary_response_box": boundary_response_box,
            "boundary_points_global": empty_coords,
            "boundary_points_local": empty_coords,
            "boundary_mask_box": torch.zeros_like(boundary_response_box, dtype=torch.bool),
        }

    keep_count = min(int(topk_points), int(valid_indices.numel()))
    top_indices_in_valid = torch.topk(flat_candidate_response[valid_indices], k=keep_count, largest=True).indices
    selected_flat_indices = valid_indices[top_indices_in_valid]

    boundary_mask_box = torch.zeros_like(boundary_response_box, dtype=torch.bool)
    boundary_mask_box.view(-1)[selected_flat_indices] = True
    boundary_points_local = torch.nonzero(boundary_mask_box, as_tuple=False).to(torch.float32)  # [N, 2]，顺序 `(y_local, x_local)`。
    boundary_points_global = boundary_points_local.clone()
    boundary_points_global[:, 0] += float(parsed_box["y_min"])
    boundary_points_global[:, 1] += float(parsed_box["x_min"])

    return {
        "boundary_response_box": boundary_response_box,
        "boundary_points_global": boundary_points_global,
        "boundary_points_local": boundary_points_local,
        "boundary_mask_box": boundary_mask_box,
    }


def estimate_local_tangent_and_normal(
    boundary_points_global: torch.Tensor,
    center_point: PointDict,
    core_ab: PointDict,
    core_ba: PointDict,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """根据中心点附近的局部伪边界点估计切线与法线。

    输入：
        boundary_points_global: Tensor, shape = (N, 2)
            局部伪边界点，全局坐标，顺序为 `(y, x)`。
        center_point: dict
            当前局部边界中心点。
        core_ab: dict
            `z_{A,B}`。
        core_ba: dict
            `z_{B,A}`。
        eps: float
            数值稳定项。

    输出：
        tangent_vector: Tensor, shape = (2,)
            局部切线方向，顺序为 `(dy, dx)`。
        normal_vector: Tensor, shape = (2,)
            局部法线方向，顺序为 `(dy, dx)`，并且方向已被校正成“从 A 指向 B”。

    说明：
        1. 优先用边界点做 PCA 拟合切线。
        2. 若边界点太少或退化，再回退到 `core_ba - core_ab` 提供的粗法线方向。
        3. 最终法线方向会根据 `core_ab -> core_ba` 的方向进行定向，不会左右颠倒。
    """
    coarse_normal = compute_local_boundary_direction(core_ab=core_ab, core_ba=core_ba, eps=eps)
    if boundary_points_global.dim() != 2 or boundary_points_global.size(-1) != 2:
        raise ValueError("boundary_points_global must have shape (N, 2)")

    if boundary_points_global.size(0) < 2:
        if coarse_normal is None:
            coarse_normal = torch.tensor([0.0, 1.0], dtype=torch.float32)  # 实在没法估时给一个稳定默认值，至少别炸。
        tangent_vector = torch.tensor([-float(coarse_normal[1]), float(coarse_normal[0])], dtype=torch.float32)
        tangent_vector = normalize_vector(tangent_vector, eps=eps)
        if tangent_vector is None:
            tangent_vector = torch.tensor([1.0, 0.0], dtype=torch.float32)
        return tangent_vector, coarse_normal

    points = boundary_points_global.to(torch.float32)
    mean_point = points.mean(dim=0, keepdim=True)
    centered_points = points - mean_point  # 这里用局部边界点自己的中心做 PCA，比硬用中心点 c 会更稳一点。

    try:
        _, _, vh = torch.linalg.svd(centered_points, full_matrices=False)
        tangent_vector = vh[0]
    except RuntimeError:
        tangent_vector = None

    tangent_vector = normalize_vector(
        torch.tensor([float(tangent_vector[0]), float(tangent_vector[1])], dtype=torch.float32)
        if tangent_vector is not None else torch.zeros(2, dtype=torch.float32),
        eps=eps,
    )

    if tangent_vector is None:
        if coarse_normal is None:
            coarse_normal = torch.tensor([0.0, 1.0], dtype=torch.float32)
        tangent_vector = torch.tensor([-float(coarse_normal[1]), float(coarse_normal[0])], dtype=torch.float32)
        tangent_vector = normalize_vector(tangent_vector, eps=eps)
        if tangent_vector is None:
            tangent_vector = torch.tensor([1.0, 0.0], dtype=torch.float32)

    normal_vector = torch.tensor(
        [-float(tangent_vector[1]), float(tangent_vector[0])],
        dtype=torch.float32,
    )  # 二维里切线旋转 90 度就是法线。
    normal_vector = normalize_vector(normal_vector, eps=eps)
    if normal_vector is None:
        normal_vector = coarse_normal if coarse_normal is not None else torch.tensor([0.0, 1.0], dtype=torch.float32)

    if coarse_normal is not None and float(torch.dot(normal_vector, coarse_normal).item()) < 0.0:
        normal_vector = -normal_vector  # 法线方向统一校正成从 A 指向 B。

    return tangent_vector, normal_vector


def generate_side_reference_centers(
    center_point: PointDict,
    normal_vector: torch.Tensor,
    box: BoxDict,
    offset_distance: float,
) -> Tuple[Dict[str, float | int], Dict[str, float | int], Optional[torch.Tensor]]:
    """由局部中心点与局部法线生成两侧粗参考中心。

    定义：
        `ref_center_a = c - offset_distance * n`
        `ref_center_b = c + offset_distance * n`

    输入：
        center_point: dict
            当前局部边界中心点。
        normal_vector: Tensor, shape = (2,)
            当前局部法线，方向约定为“从 A 指向 B”。
        box: dict
            当前局部 box。
        offset_distance: float
            沿法线两侧偏移的距离。

    输出：
        ref_center_a: dict
            A 侧粗参考中心。
        ref_center_b: dict
            对侧粗参考中心。
        used_normal: Optional[Tensor], shape = (2,)
            实际使用的法线。若法线退化，则返回 `None`。
    """
    if float(offset_distance) < 0:
        raise ValueError("offset_distance must be >= 0")

    parsed_center = _parse_point(center_point)
    if normal_vector is None:
        ref_center_a = clip_point_to_box(parsed_center, box)
        ref_center_b = clip_point_to_box(parsed_center, box)
        return ref_center_a, ref_center_b, None

    used_normal = normalize_vector(normal_vector)
    if used_normal is None:
        ref_center_a = clip_point_to_box(parsed_center, box)
        ref_center_b = clip_point_to_box(parsed_center, box)
        return ref_center_a, ref_center_b, None

    center_tensor = point_to_tensor(parsed_center)
    ref_center_a_tensor = center_tensor - float(offset_distance) * used_normal
    ref_center_b_tensor = center_tensor + float(offset_distance) * used_normal
    ref_center_a = clip_point_to_box(ref_center_a_tensor, box)
    ref_center_b = clip_point_to_box(ref_center_b_tensor, box)
    return ref_center_a, ref_center_b, used_normal


def _sample_map_nearest(
    map_2d: torch.Tensor,
    points_global_yx: torch.Tensor,
    box: BoxDict,
) -> torch.Tensor:
    """在 box 内用最近邻方式采样二维图。

    输入：
        map_2d: Tensor, shape = (h_box, w_box)
            box 内局部二维图。
        points_global_yx: Tensor, shape = (N, 2)
            全局坐标点，顺序为 `(y, x)`。
        box: dict
            当前 box。

    输出：
        sampled_values: Tensor, shape = (N,)
            对应点的采样值。
    """
    parsed_box = _parse_box(box)
    if map_2d.dim() != 2:
        raise ValueError("map_2d must have shape (h_box, w_box)")
    if points_global_yx.dim() != 2 or points_global_yx.size(-1) != 2:
        raise ValueError("points_global_yx must have shape (N, 2)")

    local_y = torch.round(points_global_yx[:, 0] - float(parsed_box["y_min"])).to(torch.long)
    local_x = torch.round(points_global_yx[:, 1] - float(parsed_box["x_min"])).to(torch.long)
    local_y = local_y.clamp(min=0, max=map_2d.size(0) - 1)
    local_x = local_x.clamp(min=0, max=map_2d.size(1) - 1)
    return map_2d[local_y, local_x]


def search_side_anchor_points(
    prob_map: torch.Tensor,
    box: BoxDict,
    boundary_response_box: torch.Tensor,
    center_point: PointDict,
    normal_vector: torch.Tensor,
    a: int,
    b: int,
    offset_distance: float,
) -> Tuple[Dict[str, float | int], Dict[str, float | int]]:
    """沿局部法线两侧搜索两个起始锚点。

    输入：
        prob_map: Tensor, shape = (B, K, H, W)
            教师模型概率图。
        box: dict
            当前局部 box。
        boundary_response_box: Tensor, shape = (h_box, w_box)
            当前 box 内的伪边界响应图。
        center_point: dict
            当前局部边界中心点。
        normal_vector: Tensor, shape = (2,)
            当前局部法线，方向为“从 A 指向 B”。
        a: int
            当前 pair 中的 A 类别。
        b: int
            当前 pair 中的对侧类别。
        offset_distance: float
            偏好锚点离中心点的距离。

    输出：
        anchor_a: dict
            A 侧起始锚点。
        anchor_b: dict
            对侧起始锚点。

    评分函数：
        `anchor_score = class_prob * boundary_penalty * distance_penalty`

    其中：
        1. `class_prob`：对应类别概率，锚点首先得像自己这一侧。
        2. `boundary_penalty`：`(1 - boundary_response)^gamma`，避免锚点直接压在强边界响应上。
        3. `distance_penalty`：偏好靠近 `offset_distance` 附近，既不能贴在中心点上，也别漂太远。
    """
    parsed_box = _parse_box(box)
    parsed_center = _parse_point(center_point)
    pa_box, pb_box = _crop_pair_probability_maps(prob_map=prob_map, box=parsed_box, a=a, b=b)

    normal_vector = normalize_vector(normal_vector)
    if normal_vector is None:
        raise ValueError("normal_vector must be a valid 2D direction")

    min_offset = max(1.0, float(offset_distance) - 1.0)
    max_offset = float(offset_distance) + float(DEFAULT_PROMPT_GENERATION_CONFIG["anchor_search_extra_distance"])
    num_samples = int(DEFAULT_PROMPT_GENERATION_CONFIG["anchor_num_samples"])
    offset_candidates = torch.linspace(min_offset, max_offset, num_samples, dtype=torch.float32)
    center_tensor = point_to_tensor(parsed_center)

    def _search_single_side(side_name: str) -> Dict[str, float | int]:
        side_sign = -1.0 if side_name == "a" else 1.0  # A 侧沿 `-n`，对侧沿 `+n`。
        target_prob_map = pa_box if side_name == "a" else pb_box

        sampled_points = center_tensor.unsqueeze(0) + side_sign * offset_candidates.unsqueeze(1) * normal_vector.unsqueeze(0)
        clipped_sampled_points = torch.stack(
            [
                torch.tensor(
                    [
                        clip_point_to_box(point=sampled_points[index], box=parsed_box)["y"],
                        clip_point_to_box(point=sampled_points[index], box=parsed_box)["x"],
                    ],
                    dtype=torch.float32,
                )
                for index in range(sampled_points.size(0))
            ],
            dim=0,
        )  # 搜索点最终全都裁回 box 内，别让法线搜索跑出去。

        class_prob = _sample_map_nearest(
            map_2d=target_prob_map,
            points_global_yx=clipped_sampled_points,
            box=parsed_box,
        )
        boundary_penalty = (
            1.0 - _sample_map_nearest(
                map_2d=boundary_response_box,
                points_global_yx=clipped_sampled_points,
                box=parsed_box,
            )
        ).clamp(min=0.0, max=1.0)
        boundary_penalty = boundary_penalty ** float(DEFAULT_PROMPT_GENERATION_CONFIG["anchor_boundary_penalty_gamma"])

        distance_sigma = float(DEFAULT_PROMPT_GENERATION_CONFIG["anchor_distance_sigma"])
        distance_penalty = torch.exp(-((offset_candidates - float(offset_distance)) ** 2) / (2.0 * distance_sigma * distance_sigma))
        anchor_score = class_prob * boundary_penalty * distance_penalty

        best_index = int(torch.argmax(anchor_score).item())
        return {
            "batch_idx": int(parsed_box["batch_idx"]),
            "y": float(clipped_sampled_points[best_index, 0].item()),
            "x": float(clipped_sampled_points[best_index, 1].item()),
            "score": float(anchor_score[best_index].item()),
            "offset": float(offset_candidates[best_index].item()),
        }

    anchor_a = _search_single_side("a")
    anchor_b = _search_single_side("b")
    return anchor_a, anchor_b


def build_side_search_windows(
    anchor_a: PointDict,
    anchor_b: PointDict,
    box: BoxDict,
    window_radius: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """根据两侧起始锚点生成局部搜索窗口。

    输入：
        anchor_a: dict
            A 侧起始锚点。
        anchor_b: dict
            对侧起始锚点。
        box: dict
            当前局部 box。
        window_radius: int
            每个锚点周围搜索窗口的半径。

    输出：
        window_a: dict
            A 侧局部搜索窗口。
        window_b: dict
            对侧局部搜索窗口。
    """
    if int(window_radius) < 0:
        raise ValueError("window_radius must be >= 0")

    parsed_box = _parse_box(box)
    parsed_anchor_a = _parse_point(anchor_a)
    parsed_anchor_b = _parse_point(anchor_b)
    radius = int(window_radius)

    center_ax = int(round(float(parsed_anchor_a["x"])))
    center_ay = int(round(float(parsed_anchor_a["y"])))
    center_bx = int(round(float(parsed_anchor_b["x"])))
    center_by = int(round(float(parsed_anchor_b["y"])))

    window_a = {
        "batch_idx": int(parsed_box["batch_idx"]),
        "x_min": max(center_ax - radius, int(parsed_box["x_min"])),
        "y_min": max(center_ay - radius, int(parsed_box["y_min"])),
        "x_max": min(center_ax + radius, int(parsed_box["x_max"])),
        "y_max": min(center_ay + radius, int(parsed_box["y_max"])),
    }
    window_b = {
        "batch_idx": int(parsed_box["batch_idx"]),
        "x_min": max(center_bx - radius, int(parsed_box["x_min"])),
        "y_min": max(center_by - radius, int(parsed_box["y_min"])),
        "x_max": min(center_bx + radius, int(parsed_box["x_max"])),
        "y_max": min(center_by + radius, int(parsed_box["y_max"])),
    }
    return _parse_box(window_a), _parse_box(window_b)


def sample_line_segment_points(
    start_point: PointDict,
    end_point: PointDict,
    sample_step: float = 1.0,
) -> torch.Tensor:
    """对两点之间的线段做离散采样。

    输入：
        start_point: dict
            线段起点。
        end_point: dict
            线段终点。
        sample_step: float
            采样步长，单位是像素。

    输出：
        sampled_points: Tensor, shape = (N, 2)
            线段上的采样点，顺序为 `(y, x)`，包含起点和终点。
    """
    if float(sample_step) <= 0:
        raise ValueError("sample_step must be > 0")

    start_tensor = point_to_tensor(start_point)
    end_tensor = point_to_tensor(end_point)
    line_length = torch.linalg.norm(end_tensor - start_tensor, ord=2)
    sample_count = max(2, int(math.ceil(float(line_length.item()) / float(sample_step))) + 1)
    interpolation_values = torch.linspace(0.0, 1.0, sample_count, dtype=torch.float32)
    sampled_points = (
        (1.0 - interpolation_values).unsqueeze(1) * start_tensor.unsqueeze(0) +
        interpolation_values.unsqueeze(1) * end_tensor.unsqueeze(0)
    )
    return sampled_points


def compute_line_boundary_consistency_score(
    anchor_point: PointDict,
    candidate_point: PointDict,
    boundary_response_box: torch.Tensor,
    box: BoxDict,
    consistency_lambda: Optional[float] = None,
) -> float:
    """计算“从锚点到候选点”的连线边界一致性分数。

    输入：
        anchor_point: dict
            当前侧的起始锚点。
        candidate_point: dict
            当前候选 prompt 点。
        boundary_response_box: Tensor, shape = (h_box, w_box)
            box 内的伪边界响应图。
        box: dict
            当前局部 box。
        consistency_lambda: Optional[float]
            一致性指数。
            若为 `None`，则使用默认超参数 `consistency_lambda`。

    输出：
        consistency_score: float
            当前候选点的连线一致性分数，范围大致在 `[0, 1]`。

    规则：
        1. 从锚点到候选点离散采样线段。
        2. 取线段上最大的边界响应。
        3. 定义：
           `consistency = (1 - max_boundary_response) ** lambda`

    说明：
        1. 如果连线穿过强边界，`max_boundary_response` 会大，一致性就会变低。
        2. 这正是本次改动最关键的抑制项，用来避免 prompt 明明在某一侧，却在连线过程中跨到了另一侧。
    """
    consistency_lambda = float(DEFAULT_PROMPT_GENERATION_CONFIG["consistency_lambda"]) if consistency_lambda is None else float(consistency_lambda)

    sampled_line = sample_line_segment_points(start_point=anchor_point, end_point=candidate_point, sample_step=1.0)
    if sampled_line.size(0) > 1:
        sampled_line = sampled_line[1:]  # 起点就是锚点本身，没必要把它也算进“是否跨边界”的惩罚里。
    if sampled_line.numel() == 0:
        return 1.0

    line_boundary_values = _sample_map_nearest(
        map_2d=boundary_response_box,
        points_global_yx=sampled_line,
        box=box,
    )
    max_boundary_response = float(line_boundary_values.max().item())
    consistency_score = max(0.0, 1.0 - max_boundary_response) ** consistency_lambda
    return float(consistency_score)


def _compute_boundary_distance_map(
    boundary_points_global: torch.Tensor,
    box: BoxDict,
) -> torch.Tensor:
    """计算当前 box 内每个像素到局部伪边界的最近距离。

    输入：
        boundary_points_global: Tensor, shape = (N, 2)
            局部伪边界点，全局坐标。
        box: dict
            当前局部 box。

    输出：
        boundary_distance_box: Tensor, shape = (h_box, w_box)
            box 内每个位置到局部伪边界的最小欧氏距离。
    """
    parsed_box = _parse_box(box)
    height_box = parsed_box["y_max"] - parsed_box["y_min"] + 1
    width_box = parsed_box["x_max"] - parsed_box["x_min"] + 1

    if boundary_points_global.numel() == 0:
        return torch.full((height_box, width_box), fill_value=1e6, dtype=torch.float32)

    y_local = torch.arange(height_box, dtype=torch.float32)
    x_local = torch.arange(width_box, dtype=torch.float32)
    grid_y_local, grid_x_local = torch.meshgrid(y_local, x_local, indexing="ij")
    grid_y_global = grid_y_local + float(parsed_box["y_min"])
    grid_x_global = grid_x_local + float(parsed_box["x_min"])

    pixel_coords_global = torch.stack([grid_y_global.reshape(-1), grid_x_global.reshape(-1)], dim=1)
    distances = torch.cdist(pixel_coords_global.to(torch.float32), boundary_points_global.to(torch.float32), p=2)
    min_distances = distances.min(dim=1).values.reshape(height_box, width_box)
    return min_distances


def _build_layer_radius_schedule(
    window_radius: int,
    num_layers: int,
) -> List[Tuple[float, float]]:
    """把总窗口半径拆成多层环带半径区间。"""
    if int(num_layers) < 1:
        raise ValueError("num_layers must be >= 1")
    if int(window_radius) < 1:
        raise ValueError("window_radius must be >= 1")

    outer_radii = []
    for layer_index in range(int(num_layers)):
        radius = max(1, int(round((layer_index + 1) * int(window_radius) / int(num_layers))))
        if len(outer_radii) == 0 or radius > outer_radii[-1]:
            outer_radii.append(radius)
    while len(outer_radii) < int(num_layers):
        outer_radii.append(outer_radii[-1] + 1)

    radius_schedule: List[Tuple[float, float]] = []
    previous_radius = 0.0
    for radius in outer_radii:
        radius_schedule.append((previous_radius, float(radius)))
        previous_radius = float(radius)
    return radius_schedule


def generate_layered_candidate_regions(
    anchor_point: PointDict,
    window: BoxDict,
    center_point: PointDict,
    normal_vector: torch.Tensor,
    boundary_distance_box: torch.Tensor,
    box: BoxDict,
    side_name: str,
    num_layers: int,
) -> List[Dict[str, object]]:
    """围绕锚点生成多层候选区域。

    输入：
        anchor_point: dict
            当前侧起始锚点。
        window: dict
            当前侧局部搜索窗口。
        center_point: dict
            当前局部边界中心点。
        normal_vector: Tensor, shape = (2,)
            当前局部法线，方向为“从 A 指向 B”。
        boundary_distance_box: Tensor, shape = (h_box, w_box)
            box 内每个位置到局部伪边界的距离。
        box: dict
            当前局部 box。
        side_name: str
            `"a"` 或 `"b"`。
        num_layers: int
            一共划多少层半径带。

    输出：
        layer_regions: list
            每一层都会返回：
            {
                "layer_index": int,
                "radius_range": (r_inner, r_outer),
                "coords_global": Tensor[M, 2],   # `(y, x)`
                "mask_window": Tensor[h_window, w_window], # bool
            }

    说明：
        1. 每一层只代表一个环带，不在这里直接选点。
        2. 候选点必须同时满足：
           - 在当前窗口内
           - 在当前侧
           - 位于边界附近窄带内
           - 落在对应的半径环带里
    """
    if side_name not in {"a", "b"}:
        raise ValueError("side_name must be either 'a' or 'b'")

    parsed_window = _parse_box(window)
    parsed_box = _parse_box(box)
    parsed_anchor = _parse_point(anchor_point)
    parsed_center = _parse_point(center_point)
    normal_vector = normalize_vector(normal_vector)
    if normal_vector is None:
        raise ValueError("normal_vector must be a valid 2D direction")

    height_window = parsed_window["y_max"] - parsed_window["y_min"] + 1
    width_window = parsed_window["x_max"] - parsed_window["x_min"] + 1
    y_global = torch.arange(parsed_window["y_min"], parsed_window["y_max"] + 1, dtype=torch.float32)
    x_global = torch.arange(parsed_window["x_min"], parsed_window["x_max"] + 1, dtype=torch.float32)
    grid_y_global, grid_x_global = torch.meshgrid(y_global, x_global, indexing="ij")

    distance_to_anchor = torch.sqrt(
        (grid_y_global - float(parsed_anchor["y"])) ** 2 +
        (grid_x_global - float(parsed_anchor["x"])) ** 2
    )
    signed_distance_to_center = (
        (grid_y_global - float(parsed_center["y"])) * float(normal_vector[0]) +
        (grid_x_global - float(parsed_center["x"])) * float(normal_vector[1])
    )  # 这个量决定某个像素到底落在法线哪一侧。

    box_local_y = (grid_y_global - float(parsed_box["y_min"])).to(torch.long)
    box_local_x = (grid_x_global - float(parsed_box["x_min"])).to(torch.long)
    boundary_distance_window = boundary_distance_box[box_local_y, box_local_x]

    side_margin = float(DEFAULT_PROMPT_GENERATION_CONFIG["normal_side_margin"])
    if side_name == "a":
        side_mask = signed_distance_to_center <= -side_margin
    else:
        side_mask = signed_distance_to_center >= side_margin

    boundary_band_mask = (
        (boundary_distance_window >= float(DEFAULT_PROMPT_GENERATION_CONFIG["boundary_distance_min"])) &
        (boundary_distance_window <= float(DEFAULT_PROMPT_GENERATION_CONFIG["boundary_distance_max"]))
    )  # prompt 只允许落在边界附近窄带内，别跑太深。

    radius_schedule = _build_layer_radius_schedule(window_radius=max(height_window, width_window) // 2, num_layers=int(num_layers))
    layer_regions: List[Dict[str, object]] = []
    for layer_index, (radius_inner, radius_outer) in enumerate(radius_schedule):
        annulus_mask = (distance_to_anchor > float(radius_inner)) & (distance_to_anchor <= float(radius_outer))
        mask_window = annulus_mask & side_mask & boundary_band_mask
        coords_local = torch.nonzero(mask_window, as_tuple=False)
        coords_global = torch.zeros((coords_local.size(0), 2), dtype=torch.float32)
        if coords_local.numel() > 0:
            coords_global[:, 0] = coords_local[:, 0].to(torch.float32) + float(parsed_window["y_min"])
            coords_global[:, 1] = coords_local[:, 1].to(torch.float32) + float(parsed_window["x_min"])

        layer_regions.append(
            {
                "layer_index": int(layer_index),
                "radius_range": (float(radius_inner), float(radius_outer)),
                "coords_global": coords_global,
                "mask_window": mask_window,
            }
        )
    return layer_regions


def _compute_point_total_score(
    prob_map: torch.Tensor,
    box: BoxDict,
    point: PointDict,
    anchor_point: PointDict,
    target_class: int,
    boundary_response_box: torch.Tensor,
    boundary_distance_box: torch.Tensor,
) -> float:
    """计算单个候选点的总分。

    总分至少包含：
        1. 类别概率项
        2. 连线边界一致性项
        3. 到伪边界的距离项

    定义：
        `total_score = class_prob * line_consistency * boundary_distance_term`
    """
    parsed_box = _parse_box(box)
    parsed_point = _parse_point(point)
    batch_idx = int(parsed_box["batch_idx"])
    _, _, height, width = prob_map.shape
    if not (0 <= int(target_class) < prob_map.size(1)):
        raise IndexError("target_class is out of range")

    y_index = int(round(float(parsed_point["y"])))
    x_index = int(round(float(parsed_point["x"])))
    y_index = min(max(y_index, 0), height - 1)
    x_index = min(max(x_index, 0), width - 1)

    class_probability = float(prob_map[batch_idx, int(target_class), y_index, x_index].item())
    line_consistency = compute_line_boundary_consistency_score(
        anchor_point=anchor_point,
        candidate_point=parsed_point,
        boundary_response_box=boundary_response_box,
        box=parsed_box,
    )

    local_y = int(round(float(parsed_point["y"]) - float(parsed_box["y_min"])))
    local_x = int(round(float(parsed_point["x"]) - float(parsed_box["x_min"])))
    local_y = min(max(local_y, 0), boundary_distance_box.size(0) - 1)
    local_x = min(max(local_x, 0), boundary_distance_box.size(1) - 1)
    boundary_distance = float(boundary_distance_box[local_y, local_x].item())

    desired_boundary_distance = float(DEFAULT_PROMPT_GENERATION_CONFIG["desired_boundary_distance"])
    boundary_distance_sigma = float(DEFAULT_PROMPT_GENERATION_CONFIG["boundary_distance_sigma"])
    boundary_distance_term = math.exp(
        -((boundary_distance - desired_boundary_distance) ** 2) /
        (2.0 * boundary_distance_sigma * boundary_distance_sigma)
    )

    total_score = class_probability * line_consistency * boundary_distance_term
    return float(total_score)


def select_best_prompt_per_layer(
    prob_map: torch.Tensor,
    box: BoxDict,
    layer_region: Dict[str, object],
    anchor_point: PointDict,
    target_class: int,
    boundary_response_box: torch.Tensor,
    boundary_distance_box: torch.Tensor,
) -> Optional[Dict[str, int | float]]:
    """在单个层内选出一个最优 prompt。

    输入：
        prob_map: Tensor, shape = (B, K, H, W)
            教师模型概率图。
        box: dict
            当前局部 box。
        layer_region: dict
            `generate_layered_candidate_regions` 的单层输出。
        anchor_point: dict
            当前侧起始锚点。
        target_class: int
            当前层要选哪一侧的 prompt。
        boundary_response_box: Tensor, shape = (h_box, w_box)
            box 内的伪边界响应图。
        boundary_distance_box: Tensor, shape = (h_box, w_box)
            box 内每个位置到伪边界的距离图。

    输出：
        best_prompt: Optional[dict]
            若当前层有可用候选点，则返回：
            {
                "batch_idx": int,
                "y": int,
                "x": int,
                "score": float,
                "layer_index": int,
            }
            若当前层没有可用点，则返回 `None`。
    """
    coords_global = layer_region["coords_global"]
    if not isinstance(coords_global, torch.Tensor):
        raise TypeError("layer_region['coords_global'] must be a torch.Tensor")
    if coords_global.numel() == 0:
        return None

    parsed_box = _parse_box(box)
    best_prompt: Optional[Dict[str, int | float]] = None
    best_score = -1.0
    for point_index in range(coords_global.size(0)):
        point = {
            "batch_idx": int(parsed_box["batch_idx"]),
            "y": float(coords_global[point_index, 0].item()),
            "x": float(coords_global[point_index, 1].item()),
        }
        total_score = _compute_point_total_score(
            prob_map=prob_map,
            box=parsed_box,
            point=point,
            anchor_point=anchor_point,
            target_class=int(target_class),
            boundary_response_box=boundary_response_box,
            boundary_distance_box=boundary_distance_box,
        )
        if total_score > best_score:
            best_score = total_score
            best_prompt = {
                "batch_idx": int(parsed_box["batch_idx"]),
                "y": int(round(float(point["y"]))),
                "x": int(round(float(point["x"]))),
                "score": float(total_score),
                "layer_index": int(layer_region["layer_index"]),
            }
    return best_prompt


def _compute_dense_score_map_for_window(
    prob_map: torch.Tensor,
    box: BoxDict,
    window: BoxDict,
    anchor_point: PointDict,
    center_point: PointDict,
    normal_vector: torch.Tensor,
    side_name: str,
    target_class: int,
    boundary_response_box: torch.Tensor,
    boundary_distance_box: torch.Tensor,
) -> torch.Tensor:
    """为可视化和调试计算当前侧窗口内的 dense score map。

    说明：
        1. 这个 dense map 主要服务于调试和兼容旧接口。
        2. 真正的 prompt 选择仍然以“每层一选”的方式进行，而不是在这个 dense map 上直接 top-k。
    """
    parsed_window = _parse_box(window)
    height_window = parsed_window["y_max"] - parsed_window["y_min"] + 1
    width_window = parsed_window["x_max"] - parsed_window["x_min"] + 1
    dense_score_map = torch.zeros((height_window, width_window), dtype=torch.float32)

    layer_regions = generate_layered_candidate_regions(
        anchor_point=anchor_point,
        window=parsed_window,
        center_point=center_point,
        normal_vector=normal_vector,
        boundary_distance_box=boundary_distance_box,
        box=box,
        side_name=side_name,
        num_layers=max(1, 3),  # dense map 只是调试图，这里固定按 3 层构一个完整候选集合。
    )
    union_mask = torch.zeros_like(dense_score_map, dtype=torch.bool)
    for layer_region in layer_regions:
        union_mask = union_mask | layer_region["mask_window"]

    coords_local = torch.nonzero(union_mask, as_tuple=False)
    parsed_box = _parse_box(box)
    for coord_index in range(coords_local.size(0)):
        local_y = int(coords_local[coord_index, 0].item())
        local_x = int(coords_local[coord_index, 1].item())
        global_point = {
            "batch_idx": int(parsed_box["batch_idx"]),
            "y": int(parsed_window["y_min"] + local_y),
            "x": int(parsed_window["x_min"] + local_x),
        }
        dense_score_map[local_y, local_x] = float(
            _compute_point_total_score(
                prob_map=prob_map,
                box=parsed_box,
                point=global_point,
                anchor_point=anchor_point,
                target_class=int(target_class),
                boundary_response_box=boundary_response_box,
                boundary_distance_box=boundary_distance_box,
            )
        )
    return dense_score_map


def _attach_prompt_roles(
    points: List[Dict[str, int | float]],
    role_for_a: str,
    role_for_b: str,
) -> List[Dict[str, int | float | str]]:
    """给一组点附加针对 A / B 两个器官的 prompt 角色。"""
    enriched_points: List[Dict[str, int | float | str]] = []
    for point in points:
        enriched_point = dict(point)
        enriched_point["role_for_a"] = str(role_for_a)
        enriched_point["role_for_b"] = str(role_for_b)
        enriched_points.append(enriched_point)
    return enriched_points


def generate_point_prompts_from_ordered_cores(
    prob_map: torch.Tensor,
    box: BoxDict,
    core_ab: PointDict,
    core_ba: PointDict,
    a: int,
    b: int,
    offset_distance: float = 3.0,
    window_radius: int = 12,
    sigma: float = 5.0,  # 当前为了兼容旧接口保留该参数，但新版逻辑不再把它作为主控项。
    topk_per_side: int = 3,
    min_distance: float = 5.0,  # 当前为了兼容旧接口保留，但新版逻辑主要靠“每层一选”而不是 NMS。
    min_score: Optional[float] = None,  # 当前为了兼容旧接口保留，若给出则会在层内最终结果上再过滤。
) -> Dict[str, object]:
    """从两个有序核心点生成单个 box 的分层、多点 point prompts。

    新版流程：
        1. 根据 `core_ab / core_ba` 计算局部中心点 `c`。
        2. 在 `c` 附近提取一小段局部伪边界。
        3. 对局部伪边界做 PCA，拟合局部切线，再得到局部法线。
        4. 沿法线两侧做一维搜索，分别找到 A 侧锚点与对侧锚点。
        5. 围绕两个锚点按分层半径构造候选环带。
        6. 在每一层中用“类别概率 + 连线边界一致性 + 边界距离项”选出一个最优 prompt。

    输入：
        prob_map: Tensor, shape = (B, K, H, W)
            教师模型概率图。
        box: dict
            当前局部 box。
        core_ab: dict
            第三部分输出的 `z_{A,B}`。
        core_ba: dict
            第三部分输出的 `z_{B,A}`。
        a: int
            当前器官对中的 A 类别 id。
        b: int
            当前器官对中的对侧类别 id。
        offset_distance: float
            从中心点沿法线两侧生成粗参考中心时的偏移距离。
        window_radius: int
            围绕两侧锚点进行分层搜索时的总半径。
        sigma: float
            为了兼容旧接口保留。
        topk_per_side: int
            新版里被解释为“每侧保留多少层 prompt”，也就是层数。
        min_distance: float
            为了兼容旧接口保留。
        min_score: Optional[float]
            若不为 `None`，则最终层内选出的 prompt 低于该值会被丢弃。

    输出：
        result_dict: dict
            兼容旧版主要输出，同时补充了大量调试字段：
            {
                "box": {...},
                "a": int,
                "b": int,
                "core_ab": {...},
                "core_ba": {...},
                "center_point": {...},
                "tangent_vector": Tensor[2],
                "direction_vector": Tensor[2],       # 这里就是最终使用的局部法线
                "normal_vector": Tensor[2],
                "ref_center_a": {...},               # 粗参考中心
                "ref_center_b": {...},
                "anchor_a": {...},                   # 真正用于分层搜索的起始锚点
                "anchor_b": {...},
                "window_a": {...},
                "window_b": {...},
                "boundary_points_global": Tensor[N, 2],
                "boundary_response_box": Tensor[h_box, w_box],
                "boundary_distance_box": Tensor[h_box, w_box],
                "layer_regions_a": [...],
                "layer_regions_b": [...],
                "score_map_a_window": Tensor[h_a, w_a],
                "score_map_b_window": Tensor[h_b, w_b],
                "points_a": [...],
                "points_b": [...],
            }
    """
    if prob_map.dim() != 4:
        raise ValueError("prob_map must have shape (B, K, H, W)")

    batch_size, _, height, width = prob_map.shape
    parsed_box = clip_box_to_image(box, height=height, width=width)
    parsed_box["a"] = int(a)
    parsed_box["b"] = int(b)
    parsed_core_ab = _parse_point(core_ab)
    parsed_core_ba = _parse_point(core_ba)

    if int(parsed_box["batch_idx"]) != int(parsed_core_ab["batch_idx"]) or int(parsed_box["batch_idx"]) != int(parsed_core_ba["batch_idx"]):
        raise ValueError("box, core_ab, and core_ba must share the same batch_idx")
    if not (0 <= int(parsed_box["batch_idx"]) < batch_size):
        raise IndexError("box batch_idx is out of range for prob_map")

    center_point = compute_center_point_from_ordered_cores(core_ab=parsed_core_ab, core_ba=parsed_core_ba)
    boundary_info = extract_local_boundary_points(
        prob_map=prob_map,
        box=parsed_box,
        a=int(a),
        b=int(b),
        center_point=center_point,
    )
    tangent_vector, normal_vector = estimate_local_tangent_and_normal(
        boundary_points_global=boundary_info["boundary_points_global"],
        center_point=center_point,
        core_ab=parsed_core_ab,
        core_ba=parsed_core_ba,
    )
    ref_center_a, ref_center_b, direction_vector = generate_side_reference_centers(
        center_point=center_point,
        normal_vector=normal_vector,
        box=parsed_box,
        offset_distance=float(offset_distance),
    )
    if direction_vector is None:
        # 真到这里说明局部法线和粗法线都退化了，那第四部分再硬搜只会越搜越偏，不如明确停住。
        raise RuntimeError("Failed to compute a valid local boundary normal for prompt generation")

    anchor_a, anchor_b = search_side_anchor_points(
        prob_map=prob_map,
        box=parsed_box,
        boundary_response_box=boundary_info["boundary_response_box"],
        center_point=center_point,
        normal_vector=direction_vector,
        a=int(a),
        b=int(b),
        offset_distance=float(offset_distance),
    )
    window_a, window_b = build_side_search_windows(
        anchor_a=anchor_a,
        anchor_b=anchor_b,
        box=parsed_box,
        window_radius=int(window_radius),
    )

    boundary_distance_box = _compute_boundary_distance_map(
        boundary_points_global=boundary_info["boundary_points_global"],
        box=parsed_box,
    )
    layer_regions_a = generate_layered_candidate_regions(
        anchor_point=anchor_a,
        window=window_a,
        center_point=center_point,
        normal_vector=direction_vector,
        boundary_distance_box=boundary_distance_box,
        box=parsed_box,
        side_name="a",
        num_layers=max(1, int(topk_per_side)),
    )
    layer_regions_b = generate_layered_candidate_regions(
        anchor_point=anchor_b,
        window=window_b,
        center_point=center_point,
        normal_vector=direction_vector,
        boundary_distance_box=boundary_distance_box,
        box=parsed_box,
        side_name="b",
        num_layers=max(1, int(topk_per_side)),
    )

    raw_points_a: List[Dict[str, int | float]] = []
    for layer_region in layer_regions_a:
        best_prompt = select_best_prompt_per_layer(
            prob_map=prob_map,
            box=parsed_box,
            layer_region=layer_region,
            anchor_point=anchor_a,
            target_class=int(a),
            boundary_response_box=boundary_info["boundary_response_box"],
            boundary_distance_box=boundary_distance_box,
        )
        if best_prompt is None:
            continue
        if min_score is not None and float(best_prompt["score"]) < float(min_score):
            continue
        raw_points_a.append(best_prompt)

    raw_points_b: List[Dict[str, int | float]] = []
    for layer_region in layer_regions_b:
        best_prompt = select_best_prompt_per_layer(
            prob_map=prob_map,
            box=parsed_box,
            layer_region=layer_region,
            anchor_point=anchor_b,
            target_class=int(b),
            boundary_response_box=boundary_info["boundary_response_box"],
            boundary_distance_box=boundary_distance_box,
        )
        if best_prompt is None:
            continue
        if min_score is not None and float(best_prompt["score"]) < float(min_score):
            continue
        raw_points_b.append(best_prompt)

    # 新版不再依赖全局 top-k + NMS；每层只取一个点，所以天然更沿边界分散。
    points_a = _attach_prompt_roles(raw_points_a, role_for_a="positive", role_for_b="negative")
    points_b = _attach_prompt_roles(raw_points_b, role_for_a="negative", role_for_b="positive")

    score_map_a_window = _compute_dense_score_map_for_window(
        prob_map=prob_map,
        box=parsed_box,
        window=window_a,
        anchor_point=anchor_a,
        center_point=center_point,
        normal_vector=direction_vector,
        side_name="a",
        target_class=int(a),
        boundary_response_box=boundary_info["boundary_response_box"],
        boundary_distance_box=boundary_distance_box,
    )
    score_map_b_window = _compute_dense_score_map_for_window(
        prob_map=prob_map,
        box=parsed_box,
        window=window_b,
        anchor_point=anchor_b,
        center_point=center_point,
        normal_vector=direction_vector,
        side_name="b",
        target_class=int(b),
        boundary_response_box=boundary_info["boundary_response_box"],
        boundary_distance_box=boundary_distance_box,
    )

    return {
        "box": parsed_box,
        "a": int(a),
        "b": int(b),
        "core_ab": parsed_core_ab,
        "core_ba": parsed_core_ba,
        "center_point": center_point,
        "tangent_vector": tangent_vector,
        "direction_vector": direction_vector,
        "normal_vector": direction_vector,
        "ref_center_a": ref_center_a,
        "ref_center_b": ref_center_b,
        "anchor_a": anchor_a,
        "anchor_b": anchor_b,
        "window_a": window_a,
        "window_b": window_b,
        "boundary_points_global": boundary_info["boundary_points_global"],
        "boundary_points_local": boundary_info["boundary_points_local"],
        "boundary_response_box": boundary_info["boundary_response_box"],
        "boundary_distance_box": boundary_distance_box,
        "layer_regions_a": layer_regions_a,
        "layer_regions_b": layer_regions_b,
        "score_map_a_window": score_map_a_window,
        "score_map_b_window": score_map_b_window,
        "points_a": points_a,
        "points_b": points_b,
    }


def generate_point_prompts_for_box_list(
    prob_map: torch.Tensor,
    box_core_list: List[Dict[str, object]],
    offset_distance: float = 3.0,
    window_radius: int = 12,
    sigma: float = 5.0,
    topk_per_side: int = 3,
    min_distance: float = 5.0,
    min_score: Optional[float] = None,
) -> List[Dict[str, object]]:
    """对多个 box 批量生成 point prompts。

    输入：
        prob_map: Tensor, shape = (B, K, H, W)
            教师模型概率图。
        box_core_list: List[dict]
            每个元素至少包含：
            - `box`
            - `core_ab`
            - `core_ba`
            - `a`
            - `b`
        其余参数保持与 `generate_point_prompts_from_ordered_cores` 一致。

    输出：
        result_list: List[dict]
            每个 box 对应一个独立结果，结构与
            `generate_point_prompts_from_ordered_cores` 的输出一致。
    """
    result_list: List[Dict[str, object]] = []
    for record in box_core_list:
        if not isinstance(record, dict):
            raise TypeError("each item in box_core_list must be a dictionary")
        if "box" not in record or "core_ab" not in record or "core_ba" not in record or "a" not in record or "b" not in record:
            raise KeyError("each item must contain 'box', 'core_ab', 'core_ba', 'a', and 'b'")

        result = generate_point_prompts_from_ordered_cores(
            prob_map=prob_map,
            box=record["box"],
            core_ab=record["core_ab"],
            core_ba=record["core_ba"],
            a=int(record["a"]),
            b=int(record["b"]),
            offset_distance=float(offset_distance),
            window_radius=int(window_radius),
            sigma=float(sigma),
            topk_per_side=int(topk_per_side),
            min_distance=float(min_distance),
            min_score=min_score,
        )
        result_list.append(result)
    return result_list


def visualize_point_prompts_in_box(
    image_or_mask_2d: torch.Tensor,
    box: BoxDict,
    core_ab: PointDict,
    core_ba: PointDict,
    ref_center_a: PointDict,
    ref_center_b: PointDict,
    points_a: List[PointDict],
    points_b: List[PointDict],
    a: int,
    b: int,
    save_path: str | Path,
    center_point: Optional[PointDict] = None,
    normal_vector: Optional[torch.Tensor] = None,
    tangent_vector: Optional[torch.Tensor] = None,
    boundary_points_global: Optional[torch.Tensor] = None,
    anchor_a: Optional[PointDict] = None,
    anchor_b: Optional[PointDict] = None,
) -> None:
    """可视化单个 box 内的 point prompt 生成结果。

    这是当前第四部分的调试函数，会尽量把下面这些元素都画出来：
        1. 当前 box
        2. 局部伪边界点
        3. 中心点 `c`
        4. 局部法线
        5. 两个有序核心点
        6. 两个锚点
        7. 每一层选中的 `points_a / points_b`

    输入：
        image_or_mask_2d: Tensor, shape = (H, W)
            用于显示的二维图像或标签图。
        box: dict
            当前局部 box。
        core_ab: dict
            `z_{A,B}` 核心点。
        core_ba: dict
            `z_{B,A}` 核心点。
        ref_center_a: dict
            A 侧粗参考中心。
        ref_center_b: dict
            对侧粗参考中心。
        points_a: list
            A 侧 prompt 点集合。
        points_b: list
            对侧 prompt 点集合。
        a: int
            当前器官对中的 A 类别 id。
        b: int
            当前器官对中的对侧类别 id。
        save_path: str | Path
            输出图像路径。
        center_point: Optional[dict]
            当前局部中心点。
        normal_vector: Optional[Tensor], shape = (2,)
            当前局部法线。
        tangent_vector: Optional[Tensor], shape = (2,)
            当前局部切线。
        boundary_points_global: Optional[Tensor], shape = (N, 2)
            当前局部伪边界点集合。
        anchor_a: Optional[dict]
            A 侧锚点。
        anchor_b: Optional[dict]
            对侧锚点。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not isinstance(image_or_mask_2d, torch.Tensor):
        image_or_mask_2d = torch.as_tensor(image_or_mask_2d)
    if image_or_mask_2d.dim() != 2:
        raise ValueError("image_or_mask_2d must have shape (H, W)")

    height, width = image_or_mask_2d.shape
    parsed_box = clip_box_to_image(box, height=height, width=width)
    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image_np = image_or_mask_2d.detach().cpu().numpy()
    use_label_cmap = not torch.is_floating_point(image_or_mask_2d)
    cmap = "nipy_spectral" if use_label_cmap else "gray"

    fig, axis = plt.subplots(1, 1, figsize=(7.4, 7.4))
    axis.imshow(image_np, cmap=cmap, interpolation="nearest")

    rect = Rectangle(
        (parsed_box["x_min"], parsed_box["y_min"]),
        parsed_box["x_max"] - parsed_box["x_min"] + 1,
        parsed_box["y_max"] - parsed_box["y_min"] + 1,
        fill=False,
        edgecolor="yellow",
        linewidth=2.0,
        linestyle="--",
    )
    axis.add_patch(rect)

    if boundary_points_global is not None and isinstance(boundary_points_global, torch.Tensor) and boundary_points_global.numel() > 0:
        axis.scatter(
            boundary_points_global[:, 1].detach().cpu().numpy(),
            boundary_points_global[:, 0].detach().cpu().numpy(),
            s=8,
            c="white",
            linewidths=0.0,
            alpha=0.45,
            label="pseudo boundary",
        )

    parsed_core_ab = _parse_point(core_ab)
    parsed_core_ba = _parse_point(core_ba)
    axis.scatter(
        [float(parsed_core_ab["x"])],
        [float(parsed_core_ab["y"])],
        s=135,
        c="red",
        marker="*",
        edgecolors="white",
        linewidths=0.8,
        label=f"core {a}->{b}",
    )
    axis.scatter(
        [float(parsed_core_ba["x"])],
        [float(parsed_core_ba["y"])],
        s=135,
        c="cyan",
        marker="*",
        edgecolors="black",
        linewidths=0.8,
        label=f"core {b}->{a}",
    )

    if center_point is not None:
        parsed_center = _parse_point(center_point)
        axis.scatter(
            [float(parsed_center["x"])],
            [float(parsed_center["y"])],
            s=80,
            c="magenta",
            marker="o",
            edgecolors="black",
            linewidths=0.7,
            label="center c",
        )

        if normal_vector is not None:
            used_normal = normalize_vector(normal_vector)
            if used_normal is not None:
                normal_scale = 8.0
                axis.arrow(
                    float(parsed_center["x"]),
                    float(parsed_center["y"]),
                    float(used_normal[1]) * normal_scale,
                    float(used_normal[0]) * normal_scale,
                    color="magenta",
                    width=0.18,
                    head_width=1.2,
                    length_includes_head=True,
                    alpha=0.90,
                )
                axis.arrow(
                    float(parsed_center["x"]),
                    float(parsed_center["y"]),
                    -float(used_normal[1]) * normal_scale,
                    -float(used_normal[0]) * normal_scale,
                    color="magenta",
                    width=0.12,
                    head_width=1.0,
                    length_includes_head=True,
                    alpha=0.60,
                )

        if tangent_vector is not None:
            used_tangent = normalize_vector(tangent_vector)
            if used_tangent is not None:
                tangent_scale = 7.0
                axis.plot(
                    [
                        float(parsed_center["x"]) - float(used_tangent[1]) * tangent_scale,
                        float(parsed_center["x"]) + float(used_tangent[1]) * tangent_scale,
                    ],
                    [
                        float(parsed_center["y"]) - float(used_tangent[0]) * tangent_scale,
                        float(parsed_center["y"]) + float(used_tangent[0]) * tangent_scale,
                    ],
                    color="magenta",
                    linestyle=":",
                    linewidth=1.4,
                    alpha=0.75,
                )

    parsed_ref_center_a = _parse_point(ref_center_a)
    parsed_ref_center_b = _parse_point(ref_center_b)
    axis.scatter(
        [float(parsed_ref_center_a["x"])],
        [float(parsed_ref_center_a["y"])],
        s=90,
        facecolors="none",
        edgecolors="orange",
        linewidths=2.0,
        marker="o",
        label="ref center A",
    )
    axis.scatter(
        [float(parsed_ref_center_b["x"])],
        [float(parsed_ref_center_b["y"])],
        s=90,
        c="lime",
        linewidths=2.0,
        marker="x",
        label="ref center B",
    )

    if anchor_a is not None:
        parsed_anchor_a = _parse_point(anchor_a)
        axis.scatter(
            [float(parsed_anchor_a["x"])],
            [float(parsed_anchor_a["y"])],
            s=100,
            c="orange",
            marker="D",
            edgecolors="black",
            linewidths=0.7,
            label="anchor A",
        )
    if anchor_b is not None:
        parsed_anchor_b = _parse_point(anchor_b)
        axis.scatter(
            [float(parsed_anchor_b["x"])],
            [float(parsed_anchor_b["y"])],
            s=100,
            c="lime",
            marker="D",
            edgecolors="black",
            linewidths=0.7,
            label="anchor B",
        )

    # 这里按 layer_index 给每一层做轻微尺寸区分，方便肉眼看 prompt 是否沿边界分散展开。
    for point in points_a:
        parsed_point = _parse_point(point)
        layer_index = int(point.get("layer_index", 0)) if isinstance(point, dict) else 0
        axis.scatter(
            [float(parsed_point["x"])],
            [float(parsed_point["y"])],
            s=72 + 18 * layer_index,
            c="orange",
            marker="o",
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
        )
        axis.text(
            float(parsed_point["x"]) + 0.8,
            float(parsed_point["y"]) + 0.8,
            f"A-L{layer_index + 1}",
            color="orange",
            fontsize=8,
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.30, "pad": 1},
        )

    for point in points_b:
        parsed_point = _parse_point(point)
        layer_index = int(point.get("layer_index", 0)) if isinstance(point, dict) else 0
        axis.scatter(
            [float(parsed_point["x"])],
            [float(parsed_point["y"])],
            s=72 + 18 * layer_index,
            c="lime",
            marker="x",
            linewidths=2.0,
            alpha=0.95,
        )
        axis.text(
            float(parsed_point["x"]) + 0.8,
            float(parsed_point["y"]) - 0.8,
            f"B-L{layer_index + 1}",
            color="lime",
            fontsize=8,
            ha="left",
            va="top",
            bbox={"facecolor": "black", "alpha": 0.30, "pad": 1},
        )

    axis.set_title(f"pair=({a},{b}), point prompt generation")
    axis.axis("off")
    axis.legend(loc="upper right", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_demo_prob_map() -> Tuple[torch.Tensor, BoxDict, PointDict, PointDict]:
    """构造最小可运行 demo 所需的 dummy 概率图、box 和两个 ordered core points。"""
    batch_size, num_classes, height, width = 1, 3, 72, 72
    prob_map = torch.zeros((batch_size, num_classes, height, width), dtype=torch.float32)

    box: BoxDict = {
        "batch_idx": 0,
        "a": 1,
        "b": 2,
        "x_min": 14,
        "y_min": 18,
        "x_max": 55,
        "y_max": 53,
    }
    core_ab: PointDict = {
        "batch_idx": 0,
        "y": 34.0,
        "x": 25.0,
        "score": 0.98,
        "ordered_boundary_key": (1, 2),
    }
    core_ba: PointDict = {
        "batch_idx": 0,
        "y": 38.0,
        "x": 41.0,
        "score": 0.97,
        "ordered_boundary_key": (2, 1),
    }

    # 这里人为造一条略带弯曲的 A-B 边界，好让“局部切线 / 法线 + 分层 prompt”这套逻辑有东西可看。
    y_grid = torch.arange(height, dtype=torch.float32).view(height, 1).expand(height, width)
    x_grid = torch.arange(width, dtype=torch.float32).view(1, width).expand(height, width)
    curved_boundary_x = 33.0 + 0.18 * (y_grid - 36.0)
    signed_distance = x_grid - curved_boundary_x

    prob_a = torch.sigmoid(-signed_distance / 2.0)
    prob_b = torch.sigmoid(signed_distance / 2.0)
    background = torch.full_like(prob_a, fill_value=0.04)

    prob_map[0, 1] = 0.08 + 0.84 * prob_a
    prob_map[0, 2] = 0.08 + 0.84 * prob_b
    prob_map[0, 0] = background
    prob_map = prob_map / prob_map.sum(dim=1, keepdim=True)
    return prob_map, box, core_ab, core_ba


def run_demo(save_dir: str | Path | None = None) -> Dict[str, object]:
    """运行第四部分的最小可执行 demo。"""
    torch.manual_seed(7)
    prob_map, box, core_ab, core_ba = _build_demo_prob_map()

    result = generate_point_prompts_from_ordered_cores(
        prob_map=prob_map,
        box=box,
        core_ab=core_ab,
        core_ba=core_ba,
        a=1,
        b=2,
        offset_distance=3.0,
        window_radius=12,
        sigma=5.0,
        topk_per_side=3,
        min_distance=5.0,
        min_score=None,
    )

    print("=" * 80)
    print("Ordered Boundary Point Prompt Demo")
    print("=" * 80)
    print(f"center_point={result['center_point']}")
    print(f"direction_vector={result['direction_vector']}")
    print(f"ref_center_a={result['ref_center_a']}")
    print(f"ref_center_b={result['ref_center_b']}")
    print(f"anchor_a={result['anchor_a']}")
    print(f"anchor_b={result['anchor_b']}")
    print(f"num_points_a={len(result['points_a'])}")
    print(f"num_points_b={len(result['points_b'])}")
    print(f"points_a={result['points_a']}")
    print(f"points_b={result['points_b']}")

    if save_dir is None:
        save_dir = Path(__file__).resolve().parent / "outputs"
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / "ordered_boundary_point_prompt_demo.png"
    visualize_point_prompts_in_box(
        image_or_mask_2d=prob_map[0, 1] - prob_map[0, 2],  # 用 A-B 概率差作底图，边界附近会更直观。
        box=box,
        core_ab=core_ab,
        core_ba=core_ba,
        ref_center_a=result["ref_center_a"],
        ref_center_b=result["ref_center_b"],
        points_a=result["points_a"],
        points_b=result["points_b"],
        a=1,
        b=2,
        save_path=save_path,
        center_point=result["center_point"],
        normal_vector=result["normal_vector"],
        tangent_vector=result["tangent_vector"],
        boundary_points_global=result["boundary_points_global"],
        anchor_a=result["anchor_a"],
        anchor_b=result["anchor_b"],
    )
    print(f"visualization saved to: {save_path}")
    return result


def main() -> None:
    """最小 demo 入口。"""
    run_demo()


if __name__ == "__main__":
    main()
