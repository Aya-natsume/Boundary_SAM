"""第四部分：基于两个有序核心边界点生成多点 SAM point prompts。

当前模块只负责一件事：
给定第三部分输出的两个有序核心边界点
1. `z_{A,B}`：最像“属于 A、邻接 B”的核心点
2. `z_{B,A}`：最像“属于 B、邻接 A”的核心点

在一个局部 box 内进一步生成真正可用于后续 SAM refinement 的多点 point prompts。

当前设计遵循下面这条主线：
1. 先由 `z_{A,B}` 和 `z_{B,A}` 推导局部跨边界方向。
2. 再分别向 A 内部和 B 内部轻微偏移，得到两侧参考中心。
3. 然后在两侧参考中心附近建立局部搜索窗口。
4. 最后基于“概率差值 + 距离衰减”分别选出多点、空间分散的支持点。

注意：
1. 当前模块不生成 box prompt。
2. 当前模块不直接调用 SAM。
3. 当前模块不重新引入 `pred_mask == A/B` 的硬划分逻辑。
4. 当前模块重点是把 point prompt 生成逻辑做得清楚、稳定、可解释。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


PointDict = Dict[str, int | float | str | Tuple[int, int]]
BoxDict = Dict[str, int | float | Tuple[int, int, int, int]]


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
                "a": int,      # 若原始输入里没有，则默认 -1
                "b": int,      # 若原始输入里没有，则默认 -1
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
            - `score`
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
            坐标顺序固定为：
            - `coord_tensor[0] = y`
            - `coord_tensor[1] = x`
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
            若向量长度过小，则返回 `None`。
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
            与 `_parse_box` 的输出格式一致，但坐标被裁剪到了：
            - `0 <= x_min <= x_max < W`
            - `0 <= y_min <= y_max < H`
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
    """根据两个有序核心点计算局部跨边界方向。

    定义：
        `n = (z_{B,A} - z_{A,B}) / (||z_{B,A} - z_{A,B}|| + eps)`

    输入：
        core_ab: dict
            第三部分输出的 `z_{A,B}`。
        core_ba: dict
            第三部分输出的 `z_{B,A}`。
        eps: float
            数值稳定项。

    输出：
        direction_vector: Optional[Tensor], shape = (2,)
            二维单位向量，顺序为 `(dy, dx)`。
            若两个点重合或过近，返回 `None`。

    说明：
        1. 这里返回 `None` 而不是硬塞一个默认方向。
        2. 因为当两个 anchor 本身没有稳定几何间距时，强行指定方向反而容易把后续点搜索带偏。
        3. 后续 `generate_side_reference_centers` 会在 `None` 的情况下退化成“不偏移，只围绕 anchor 本身建窗口”。
    """
    point_ab = point_to_tensor(core_ab)
    point_ba = point_to_tensor(core_ba)
    delta = point_ba - point_ab
    return normalize_vector(delta, eps=eps)


def generate_side_reference_centers(
    core_ab: PointDict,
    core_ba: PointDict,
    box: BoxDict,
    offset_distance: float,
) -> Tuple[Dict[str, float | int], Dict[str, float | int], Optional[torch.Tensor]]:
    """根据两个有序核心点生成两侧参考中心。

    定义：
        `ref_center_a = z_{A,B} - offset_distance * n`
        `ref_center_b = z_{B,A} + offset_distance * n`

    输入：
        core_ab: dict
            `z_{A,B}` 核心点。
        core_ba: dict
            `z_{B,A}` 核心点。
        box: dict
            当前局部框，闭区间坐标。
        offset_distance: float
            沿局部跨边界方向向两侧内部偏移的距离。

    输出：
        ref_center_a: dict
            A 侧参考中心，坐标允许是浮点数：
            {
                "batch_idx": int,
                "y": float,
                "x": float,
            }
        ref_center_b: dict
            B 侧参考中心，坐标允许是浮点数。
        direction_vector: Optional[Tensor], shape = (2,)
            局部跨边界单位方向 `(dy, dx)`。

    说明：
        1. 若 `direction_vector is None`，则退化为：
           - `ref_center_a = z_{A,B}`
           - `ref_center_b = z_{B,A}`
        2. 偏移后的参考中心若超出 box，会被裁剪回 box 内。
    """
    if float(offset_distance) < 0:
        raise ValueError("offset_distance must be >= 0")

    parsed_core_ab = _parse_point(core_ab)
    parsed_core_ba = _parse_point(core_ba)
    if int(parsed_core_ab["batch_idx"]) != int(parsed_core_ba["batch_idx"]):
        raise ValueError("core_ab and core_ba must belong to the same batch index")

    direction_vector = compute_local_boundary_direction(core_ab=core_ab, core_ba=core_ba)
    point_ab = point_to_tensor(parsed_core_ab)
    point_ba = point_to_tensor(parsed_core_ba)

    if direction_vector is None:
        ref_center_a = clip_point_to_box(parsed_core_ab, box)
        ref_center_b = clip_point_to_box(parsed_core_ba, box)
        return ref_center_a, ref_center_b, None

    ref_center_a_tensor = point_ab - float(offset_distance) * direction_vector
    ref_center_b_tensor = point_ba + float(offset_distance) * direction_vector

    ref_center_a = clip_point_to_box(ref_center_a_tensor, box)
    ref_center_b = clip_point_to_box(ref_center_b_tensor, box)
    return ref_center_a, ref_center_b, direction_vector


def build_side_search_windows(
    ref_center_a: PointDict,
    ref_center_b: PointDict,
    box: BoxDict,
    window_radius: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """根据两侧参考中心生成局部搜索窗口。

    输入：
        ref_center_a: dict
            A 侧参考中心，至少包含：
            - `batch_idx`
            - `y`
            - `x`
        ref_center_b: dict
            B 侧参考中心。
        box: dict
            当前局部 box。
        window_radius: int
            以参考中心为中心，在 `x/y` 两个方向各扩展多少像素。

    输出：
        window_a: dict
            A 侧局部窗口：
            {
                "batch_idx": int,
                "x_min": int,
                "y_min": int,
                "x_max": int,
                "y_max": int,
            }
        window_b: dict
            B 侧局部窗口，格式同上。

    说明：
        1. 窗口坐标最终一定会被裁剪到当前 box 内。
        2. 这里的窗口仍然是闭区间坐标。
    """
    if int(window_radius) < 0:
        raise ValueError("window_radius must be >= 0")

    parsed_box = _parse_box(box)
    parsed_ref_a = _parse_point(ref_center_a)
    parsed_ref_b = _parse_point(ref_center_b)
    radius = int(window_radius)

    center_ax = int(round(float(parsed_ref_a["x"])))
    center_ay = int(round(float(parsed_ref_a["y"])))
    center_bx = int(round(float(parsed_ref_b["x"])))
    center_by = int(round(float(parsed_ref_b["y"])))

    window_a = {
        "batch_idx": int(parsed_box["batch_idx"]),
        "x_min": center_ax - radius,
        "y_min": center_ay - radius,
        "x_max": center_ax + radius,
        "y_max": center_ay + radius,
    }
    window_b = {
        "batch_idx": int(parsed_box["batch_idx"]),
        "x_min": center_bx - radius,
        "y_min": center_by - radius,
        "x_max": center_bx + radius,
        "y_max": center_by + radius,
    }

    # 先裁到当前 box 内，而不是直接裁到整张图里。第四部分不该越过本地 box 边界。
    window_a["x_min"] = max(int(window_a["x_min"]), int(parsed_box["x_min"]))
    window_a["y_min"] = max(int(window_a["y_min"]), int(parsed_box["y_min"]))
    window_a["x_max"] = min(int(window_a["x_max"]), int(parsed_box["x_max"]))
    window_a["y_max"] = min(int(window_a["y_max"]), int(parsed_box["y_max"]))

    window_b["x_min"] = max(int(window_b["x_min"]), int(parsed_box["x_min"]))
    window_b["y_min"] = max(int(window_b["y_min"]), int(parsed_box["y_min"]))
    window_b["x_max"] = min(int(window_b["x_max"]), int(parsed_box["x_max"]))
    window_b["y_max"] = min(int(window_b["y_max"]), int(parsed_box["y_max"]))

    return _parse_box(window_a), _parse_box(window_b)


def _compute_distance_decay(
    window: BoxDict,
    ref_center: PointDict,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    """在给定窗口内计算相对参考中心的高斯距离衰减。"""
    parsed_window = _parse_box(window)
    parsed_ref_center = _parse_point(ref_center)
    if float(sigma) <= 0:
        raise ValueError("sigma must be > 0")

    ys = torch.arange(parsed_window["y_min"], parsed_window["y_max"] + 1, dtype=torch.float32, device=device)
    xs = torch.arange(parsed_window["x_min"], parsed_window["x_max"] + 1, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    distance_squared = (grid_y - float(parsed_ref_center["y"])) ** 2 + (grid_x - float(parsed_ref_center["x"])) ** 2
    return torch.exp(-distance_squared / (2.0 * float(sigma) * float(sigma)))


def compute_side_prompt_score_maps(
    prob_map: torch.Tensor,
    box: BoxDict,
    window_a: BoxDict,
    window_b: BoxDict,
    ref_center_a: PointDict,
    ref_center_b: PointDict,
    a: int,
    b: int,
    sigma: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """分别在 A 侧和 B 侧局部窗口中计算多点 prompt 的局部评分图。

    A 侧分数定义：
        `s_A(x) = (p_A(x) - p_B(x)) * exp(-||x-ref_center_a||^2 / (2*sigma^2))`

    B 侧分数定义：
        `s_B(x) = (p_B(x) - p_A(x)) * exp(-||x-ref_center_b||^2 / (2*sigma^2))`

    输入：
        prob_map: Tensor, shape = (B, K, H, W)
            教师模型的类别概率图。
        box: dict
            当前局部 box。
        window_a: dict
            A 侧局部窗口。
        window_b: dict
            B 侧局部窗口。
        ref_center_a: dict
            A 侧参考中心。
        ref_center_b: dict
            B 侧参考中心。
        a: int
            当前器官对中的 A 类别 id。
        b: int
            当前器官对中的 B 类别 id。
        sigma: float
            距离衰减的高斯尺度。

    输出：
        score_map_a_window: Tensor, shape = (h_a, w_a)
            A 侧局部 prompt score map。
        score_map_b_window: Tensor, shape = (h_b, w_b)
            B 侧局部 prompt score map。

    说明：
        1. 这里不引入 `pred_mask == A/B` 的硬约束。
        2. 这里不引入额外 soft constraint。
        3. 当前阶段只保留：
           - 概率差值项
           - 到参考中心的距离衰减项
    """
    if prob_map.dim() != 4:
        raise ValueError("prob_map must have shape (B, K, H, W)")

    batch_size, num_classes, height, width = prob_map.shape
    if not (0 <= int(a) < num_classes and 0 <= int(b) < num_classes):
        raise IndexError("a and b must be valid class indices for prob_map")

    parsed_box = clip_box_to_image(box, height=height, width=width)
    parsed_window_a = clip_box_to_image(window_a, height=height, width=width)
    parsed_window_b = clip_box_to_image(window_b, height=height, width=width)
    batch_idx = int(parsed_box["batch_idx"])

    if not (0 <= batch_idx < batch_size):
        raise IndexError("box batch_idx is out of range")

    pa_window_a = prob_map[
        batch_idx,
        int(a),
        parsed_window_a["y_min"]:parsed_window_a["y_max"] + 1,
        parsed_window_a["x_min"]:parsed_window_a["x_max"] + 1,
    ].to(torch.float32)
    pb_window_a = prob_map[
        batch_idx,
        int(b),
        parsed_window_a["y_min"]:parsed_window_a["y_max"] + 1,
        parsed_window_a["x_min"]:parsed_window_a["x_max"] + 1,
    ].to(torch.float32)
    pa_window_b = prob_map[
        batch_idx,
        int(a),
        parsed_window_b["y_min"]:parsed_window_b["y_max"] + 1,
        parsed_window_b["x_min"]:parsed_window_b["x_max"] + 1,
    ].to(torch.float32)
    pb_window_b = prob_map[
        batch_idx,
        int(b),
        parsed_window_b["y_min"]:parsed_window_b["y_max"] + 1,
        parsed_window_b["x_min"]:parsed_window_b["x_max"] + 1,
    ].to(torch.float32)

    decay_a = _compute_distance_decay(
        window=parsed_window_a,
        ref_center=ref_center_a,
        sigma=float(sigma),
        device=prob_map.device,
    )
    decay_b = _compute_distance_decay(
        window=parsed_window_b,
        ref_center=ref_center_b,
        sigma=float(sigma),
        device=prob_map.device,
    )

    score_map_a_window = (pa_window_a - pb_window_a) * decay_a
    score_map_b_window = (pb_window_b - pa_window_b) * decay_b
    return score_map_a_window, score_map_b_window


def select_multiple_prompt_points(
    score_map_window: torch.Tensor,
    window: BoxDict,
    topk: int = 3,
    min_distance: float = 5.0,
    min_score: Optional[float] = None,
) -> List[Dict[str, int | float]]:
    """从单侧局部 score map 中选出多个空间分散的高质量点。

    输入：
        score_map_window: Tensor, shape = (h_window, w_window)
            某一侧窗口内的局部评分图。
        window: dict
            当前局部窗口，闭区间坐标。
        topk: int
            最多保留多少个点。
        min_distance: float
            两个候选点之间允许的最小欧氏距离。
        min_score: Optional[float]
            若不为 `None`，则先过滤掉低于该分数的像素。

    输出：
        selected_points: List[dict]
            每个点包含：
            - `batch_idx`
            - `y`
            - `x`
            - `score`

    说明：
        1. 这里不是简单地取 top-k 最大值。
        2. 当前采用“分数排序 + 最小距离抑制”的方式，避免点扎堆。
    """
    if topk < 1:
        raise ValueError("topk must be >= 1")
    if float(min_distance) < 0:
        raise ValueError("min_distance must be >= 0")
    if score_map_window.dim() != 2:
        raise ValueError("score_map_window must have shape (h_window, w_window)")

    parsed_window = _parse_box(window)
    expected_height = parsed_window["y_max"] - parsed_window["y_min"] + 1
    expected_width = parsed_window["x_max"] - parsed_window["x_min"] + 1
    if tuple(score_map_window.shape) != (expected_height, expected_width):
        raise ValueError("score_map_window shape does not match the provided window")

    flat_scores = score_map_window.reshape(-1)
    valid_mask = torch.isfinite(flat_scores)
    if min_score is not None:
        valid_mask = valid_mask & (flat_scores >= float(min_score))

    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    if valid_indices.numel() == 0:
        return []

    sorted_order = torch.argsort(flat_scores[valid_indices], descending=True)
    sorted_flat_indices = valid_indices[sorted_order]

    selected_points: List[Dict[str, int | float]] = []
    selected_local_coords: List[Tuple[int, int]] = []
    for flat_index in sorted_flat_indices.tolist():
        local_y = int(flat_index // expected_width)
        local_x = int(flat_index % expected_width)

        if selected_local_coords:
            should_skip = False
            for chosen_y, chosen_x in selected_local_coords:
                distance = ((local_y - chosen_y) ** 2 + (local_x - chosen_x) ** 2) ** 0.5
                if distance < float(min_distance):
                    should_skip = True
                    break
            if should_skip:
                continue

        global_y = int(parsed_window["y_min"] + local_y)
        global_x = int(parsed_window["x_min"] + local_x)
        selected_points.append(
            {
                "batch_idx": int(parsed_window["batch_idx"]),
                "y": global_y,
                "x": global_x,
                "score": float(score_map_window[local_y, local_x].item()),
            }
        )
        selected_local_coords.append((local_y, local_x))
        if len(selected_points) >= int(topk):
            break

    return selected_points


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
    sigma: float = 5.0,
    topk_per_side: int = 3,
    min_distance: float = 5.0,
    min_score: Optional[float] = None,
) -> Dict[str, object]:
    """从两个有序核心点生成单个 box 的多点 point prompts。

    流程：
        1. 根据 `core_ab` 和 `core_ba` 计算局部跨边界方向。
        2. 生成 A 侧和 B 侧参考中心。
        3. 构造两侧局部搜索窗口。
        4. 计算两侧局部 score map。
        5. 在两侧分别做“分数排序 + 最小距离抑制”，选出多个空间分散的高质量点。

    输入：
        prob_map: Tensor, shape = (B, K, H, W)
            教师模型的目标域概率图。
        box: dict
            当前局部 box，标准格式建议为：
            {
                "batch_idx": int,
                "a": int,
                "b": int,
                "x_min": int,
                "y_min": int,
                "x_max": int,
                "y_max": int,
            }
        core_ab: dict
            第三部分输出的 `z_{A,B}`。
        core_ba: dict
            第三部分输出的 `z_{B,A}`。
        a: int
            当前器官对中的 A 类别 id。
        b: int
            当前器官对中的 B 类别 id。
        offset_distance: float
            从核心点向器官内部偏移的距离。
        window_radius: int
            两侧局部搜索窗口半径。
        sigma: float
            距离衰减的高斯尺度。
        topk_per_side: int
            每一侧最多保留多少个 prompt 点。
        min_distance: float
            同一侧不同点之间的最小欧氏距离。
        min_score: Optional[float]
            低于该分数的点会先被过滤。

    输出：
        result_dict: dict
            {
                "box": {...},
                "a": int,
                "b": int,
                "core_ab": {...},
                "core_ba": {...},
                "direction_vector": Tensor[2] or None,
                "ref_center_a": {...},
                "ref_center_b": {...},
                "window_a": {...},
                "window_b": {...},
                "score_map_a_window": Tensor[h_a, w_a],
                "score_map_b_window": Tensor[h_b, w_b],
                "points_a": [...],
                "points_b": [...],
            }

    说明：
        1. `points_a` 表示更靠 A 一侧的支持点。
        2. `points_b` 表示更靠 B 一侧的支持点。
        3. 对器官 A 的 refinement：
           - `points_a` 是 positive
           - `points_b` 是 negative
        4. 对器官 B 的 refinement，角色正好相反。
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

    ref_center_a, ref_center_b, direction_vector = generate_side_reference_centers(
        core_ab=parsed_core_ab,
        core_ba=parsed_core_ba,
        box=parsed_box,
        offset_distance=float(offset_distance),
    )
    window_a, window_b = build_side_search_windows(
        ref_center_a=ref_center_a,
        ref_center_b=ref_center_b,
        box=parsed_box,
        window_radius=int(window_radius),
    )
    score_map_a_window, score_map_b_window = compute_side_prompt_score_maps(
        prob_map=prob_map,
        box=parsed_box,
        window_a=window_a,
        window_b=window_b,
        ref_center_a=ref_center_a,
        ref_center_b=ref_center_b,
        a=int(a),
        b=int(b),
        sigma=float(sigma),
    )

    raw_points_a = select_multiple_prompt_points(
        score_map_window=score_map_a_window,
        window=window_a,
        topk=int(topk_per_side),
        min_distance=float(min_distance),
        min_score=min_score,
    )
    raw_points_b = select_multiple_prompt_points(
        score_map_window=score_map_b_window,
        window=window_b,
        topk=int(topk_per_side),
        min_distance=float(min_distance),
        min_score=min_score,
    )

    points_a = _attach_prompt_roles(raw_points_a, role_for_a="positive", role_for_b="negative")
    points_b = _attach_prompt_roles(raw_points_b, role_for_a="negative", role_for_b="positive")

    return {
        "box": parsed_box,
        "a": int(a),
        "b": int(b),
        "core_ab": parsed_core_ab,
        "core_ba": parsed_core_ba,
        "direction_vector": direction_vector,
        "ref_center_a": ref_center_a,
        "ref_center_b": ref_center_b,
        "window_a": window_a,
        "window_b": window_b,
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
        offset_distance: float
            参考中心偏移距离。
        window_radius: int
            两侧搜索窗口半径。
        sigma: float
            距离衰减尺度。
        topk_per_side: int
            每侧最多选多少个点。
        min_distance: float
            同侧点之间的最小距离。
        min_score: Optional[float]
            低于该分数的点会先被过滤。

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
) -> None:
    """可视化单个 box 内的 point prompt 生成结果。

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
            A 侧参考中心。
        ref_center_b: dict
            B 侧参考中心。
        points_a: list
            A 侧支持点集合。
        points_b: list
            B 侧支持点集合。
        a: int
            当前器官对中的 A 类别 id。
        b: int
            当前器官对中的 B 类别 id。
        save_path: str | Path
            输出图像路径。
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

    fig, axis = plt.subplots(1, 1, figsize=(7.0, 7.0))
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

    parsed_core_ab = _parse_point(core_ab)
    parsed_core_ba = _parse_point(core_ba)
    parsed_ref_center_a = _parse_point(ref_center_a)
    parsed_ref_center_b = _parse_point(ref_center_b)

    # 两个 ordered core points 只是 boundary anchors，所以单独用醒目的星号标出来。
    axis.scatter(
        [float(parsed_core_ab["x"])],
        [float(parsed_core_ab["y"])],
        s=130,
        c="red",
        marker="*",
        edgecolors="white",
        linewidths=0.8,
        label=f"core {a}->{b}",
    )
    axis.scatter(
        [float(parsed_core_ba["x"])],
        [float(parsed_core_ba["y"])],
        s=130,
        c="cyan",
        marker="*",
        edgecolors="black",
        linewidths=0.8,
        label=f"core {b}->{a}",
    )

    # 参考中心只是局部搜索的重心，不是最终 prompt，所以用空心圆和叉号区分出来。
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

    for point in points_a:
        parsed_point = _parse_point(point)
        axis.scatter(
            [float(parsed_point["x"])],
            [float(parsed_point["y"])],
            s=75,
            c="orange",
            marker="o",
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
        )

    for point in points_b:
        parsed_point = _parse_point(point)
        axis.scatter(
            [float(parsed_point["x"])],
            [float(parsed_point["y"])],
            s=75,
            c="lime",
            marker="x",
            linewidths=2.0,
            alpha=0.95,
        )

    axis.set_title(f"pair=({a},{b}), point prompt generation")
    axis.axis("off")
    axis.legend(loc="upper right", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_demo_prob_map() -> Tuple[torch.Tensor, BoxDict, PointDict, PointDict]:
    """构造最小可运行 demo 所需的 dummy 概率图、box 和两个 ordered core points。"""
    batch_size, num_classes, height, width = 1, 3, 64, 64
    prob_map = torch.zeros((batch_size, num_classes, height, width), dtype=torch.float32)

    # 先给背景一个小底值，避免概率全零看起来太假。
    prob_map[:, 0] = 0.05
    prob_map[:, 1] = 0.15
    prob_map[:, 2] = 0.15

    box: BoxDict = {
        "batch_idx": 0,
        "a": 1,
        "b": 2,
        "x_min": 16,
        "y_min": 20,
        "x_max": 47,
        "y_max": 43,
    }

    core_ab: PointDict = {
        "batch_idx": 0,
        "y": 31,
        "x": 26,
        "score": 0.98,
        "ordered_boundary_key": (1, 2),
    }
    core_ba: PointDict = {
        "batch_idx": 0,
        "y": 31,
        "x": 38,
        "score": 0.97,
        "ordered_boundary_key": (2, 1),
    }

    # 在 box 左半边人为构造 A 更强、右半边 B 更强的概率分布。
    ys = torch.arange(height, dtype=torch.float32).view(height, 1).expand(height, width)
    xs = torch.arange(width, dtype=torch.float32).view(1, width).expand(height, width)

    center_a = torch.tensor([31.0, 23.0], dtype=torch.float32)
    center_b = torch.tensor([31.0, 41.0], dtype=torch.float32)
    gauss_a = torch.exp(-((ys - center_a[0]) ** 2 + (xs - center_a[1]) ** 2) / (2.0 * 6.0 * 6.0))
    gauss_b = torch.exp(-((ys - center_b[0]) ** 2 + (xs - center_b[1]) ** 2) / (2.0 * 6.0 * 6.0))

    prob_map[0, 1] = 0.12 + 0.75 * gauss_a + 0.05 * gauss_b
    prob_map[0, 2] = 0.12 + 0.05 * gauss_a + 0.75 * gauss_b
    prob_map[0, 0] = 1.0 - (prob_map[0, 1] + prob_map[0, 2])
    prob_map = torch.clamp(prob_map, min=1e-4)
    prob_map = prob_map / prob_map.sum(dim=1, keepdim=True)
    return prob_map, box, core_ab, core_ba


def run_demo(save_dir: str | Path | None = None) -> Dict[str, object]:
    """运行第四部分的最小可执行 demo。

    demo 会完成：
        1. 构造 dummy `prob_map`
        2. 构造一个 dummy `box`
        3. 构造两个 dummy ordered core points
        4. 跑通 `generate_point_prompts_from_ordered_cores`
        5. 打印：
           - `direction_vector`
           - `ref_center_a / ref_center_b`
           - `points_a / points_b` 数量
        6. 保存一张可视化图
    """
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
    print(f"direction_vector={result['direction_vector']}")
    print(f"ref_center_a={result['ref_center_a']}")
    print(f"ref_center_b={result['ref_center_b']}")
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
        image_or_mask_2d=prob_map[0, 1] - prob_map[0, 2],  # 用 A-B 概率差作底图，边界两侧会更清楚。
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
    )
    print(f"visualization saved to: {save_path}")
    return result


def main() -> None:
    """最小 demo 入口。"""
    run_demo()


if __name__ == "__main__":
    main()
