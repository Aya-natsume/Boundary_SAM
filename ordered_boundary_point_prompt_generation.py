"""第四部分：在第三步条带框内沿边界均匀分段生成 SAM point prompts。

当前新版第四部分只做下面这些事：
1. 接收第三步构造好的条带框几何结果；
2. 沿条带框的切线方向，把整条边界段平均切成 `n` 个小段；
3. 每个小段都覆盖完整法线厚度，不再做二维网格切块；
4. 在每个小段内，分别为 A 侧和 B 侧各选一个 ordered prototype 响应最高的点；
5. 输出最终可直接送给 SAM 的 positive / negative point 集合。

注意：
1. 当前第四部分不再沿用旧版“在二维窗口里直接 top-k 取点”的逻辑。
2. 当前第四部分不再依赖 core 点做复杂推导；条带框几何已经在第三步里定好了。
3. 为了兼容旧调用接口，函数名仍然保留 `generate_point_prompts_from_ordered_cores`，
   但它现在真正消费的是第三步放进 `box` 里的条带框几何与 ordered response maps。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


PointDict = Dict[str, object]
BoxDict = Dict[str, object]


DEFAULT_SEGMENT_PROMPT_CONFIG: Dict[str, float | int | bool] = {
    "center_exclusion_ratio": 0.12,
    "use_probability_weight": False,
}
# 第四步这里只负责“在条带框里均匀取点”，所以参数故意保持很少；
# 和条带框构造相关的几何超参数应当留在第三步，别再把职责缠回去。


def _parse_box(box: BoxDict) -> Dict[str, int]:
    """把 box 解析成统一闭区间格式。"""
    if not isinstance(box, dict):
        raise TypeError("box must be a dictionary")
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
    return {
        "batch_idx": int(box.get("batch_idx", 0)),
        "x_min": int(x_min),
        "y_min": int(y_min),
        "x_max": int(x_max),
        "y_max": int(y_max),
    }


def _parse_point(point: PointDict) -> Dict[str, float | int]:
    """把点字典解析成统一格式。"""
    if not isinstance(point, dict):
        raise TypeError("point must be a dictionary")
    if "batch_idx" not in point or "y" not in point or "x" not in point:
        raise KeyError("point must contain 'batch_idx', 'y', and 'x'")
    parsed_point: Dict[str, float | int] = {
        "batch_idx": int(point["batch_idx"]),
        "y": float(point["y"]),
        "x": float(point["x"]),
    }
    if "score" in point:
        parsed_point["score"] = float(point["score"])
    return parsed_point


def point_to_tensor(point: PointDict) -> torch.Tensor:
    """把点字典转换成 `(y, x)` 张量。"""
    parsed_point = _parse_point(point)
    return torch.tensor([float(parsed_point["y"]), float(parsed_point["x"])], dtype=torch.float32)


def normalize_vector(v: torch.Tensor, eps: float = 1e-6) -> Optional[torch.Tensor]:
    """对二维向量做单位化。"""
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
    """把点裁剪到当前 box 内。"""
    parsed_box = _parse_box(box)
    if isinstance(point, torch.Tensor):
        if point.numel() != 2:
            raise ValueError("point tensor must have shape (2,)")
        y = float(point.reshape(2)[0].item())
        x = float(point.reshape(2)[1].item())
        batch_idx = int(parsed_box["batch_idx"])
    else:
        parsed_point = _parse_point(point)
        y = float(parsed_point["y"])
        x = float(parsed_point["x"])
        batch_idx = int(parsed_point["batch_idx"])
    return {
        "batch_idx": batch_idx,
        "y": min(max(y, float(parsed_box["y_min"])), float(parsed_box["y_max"])),
        "x": min(max(x, float(parsed_box["x_min"])), float(parsed_box["x_max"])),
    }


def clip_box_to_image(box: BoxDict, height: int, width: int) -> Dict[str, int]:
    """把 box 裁剪到图像边界内。"""
    parsed_box = _parse_box(box)
    parsed_box["x_min"] = max(0, min(parsed_box["x_min"], int(width) - 1))
    parsed_box["x_max"] = max(0, min(parsed_box["x_max"], int(width) - 1))
    parsed_box["y_min"] = max(0, min(parsed_box["y_min"], int(height) - 1))
    parsed_box["y_max"] = max(0, min(parsed_box["y_max"], int(height) - 1))
    if parsed_box["x_max"] < parsed_box["x_min"]:
        parsed_box["x_max"] = parsed_box["x_min"]
    if parsed_box["y_max"] < parsed_box["y_min"]:
        parsed_box["y_max"] = parsed_box["y_min"]
    return parsed_box


def compute_local_boundary_direction(core_ab: PointDict, core_ba: PointDict, eps: float = 1e-6) -> Optional[torch.Tensor]:
    """兼容旧接口：由两侧截止点粗略估计跨边界方向。"""
    delta = point_to_tensor(core_ba) - point_to_tensor(core_ab)
    return normalize_vector(delta, eps=eps)


def _extract_required_strip_data(box: BoxDict) -> Dict[str, object]:
    """从第三步输出的 box 中取出第四步真正需要的条带框数据。"""
    # 这里明确要求第三步已经把条带框几何和 ordered 响应算完；
    # 第四步只消费这些结果，不再重复推导法线、厚度或局部边界。
    required_tensor_keys = ["response_a_box", "response_b_box", "tangent_coord_box", "normal_coord_box", "strip_mask_box"]
    missing_keys = [key for key in required_tensor_keys if key not in box]
    if missing_keys:
        raise KeyError(f"box is missing required strip fields: {missing_keys}")
    required_meta_keys = ["strip_center_y", "strip_center_x", "strip_half_length", "strip_half_width", "tangent_vector", "normal_vector"]
    missing_meta = [key for key in required_meta_keys if key not in box]
    if missing_meta:
        raise KeyError(f"box is missing required strip geometry fields: {missing_meta}")
    return {
        "response_a_box": torch.as_tensor(box["response_a_box"], dtype=torch.float32),
        "response_b_box": torch.as_tensor(box["response_b_box"], dtype=torch.float32),
        "tangent_coord_box": torch.as_tensor(box["tangent_coord_box"], dtype=torch.float32),
        "normal_coord_box": torch.as_tensor(box["normal_coord_box"], dtype=torch.float32),
        "strip_mask_box": torch.as_tensor(box["strip_mask_box"]).bool(),
        "tangent_vector": torch.as_tensor(box["tangent_vector"], dtype=torch.float32),
        "normal_vector": torch.as_tensor(box["normal_vector"], dtype=torch.float32),
        "strip_center_y": float(box["strip_center_y"]),
        "strip_center_x": float(box["strip_center_x"]),
        "strip_half_length": float(box["strip_half_length"]),
        "strip_half_width": float(box["strip_half_width"]),
    }


def split_strip_box_along_tangent(box: BoxDict, num_segments: int) -> List[Dict[str, object]]:
    """沿条带框切线方向平均切成 `num_segments` 段。

    输入：
        box: dict
            第三步输出的条带框 box，内部必须已经带有：
            - `tangent_coord_box`
            - `strip_mask_box`
            - `strip_half_length`
        num_segments: int
            切成多少段。

    输出：
        segment_list: list
            每个元素都描述一段切线区间：
            {
                "segment_index": int,
                "t_min": float,
                "t_max": float,
                "mask": Tensor[h_box, w_box],
            }
    """
    if num_segments < 1:
        raise ValueError("num_segments must be >= 1")
    strip_data = _extract_required_strip_data(box)
    tangent_coord_box = strip_data["tangent_coord_box"]
    strip_mask_box = strip_data["strip_mask_box"]
    half_length = float(strip_data["strip_half_length"])

    # 分段只沿切线方向做，法线厚度保持整段共享；
    # 这正是新版第四步和旧版“二维 top-k 取点”最本质的差别。
    edges = torch.linspace(-half_length, half_length, steps=int(num_segments) + 1, dtype=torch.float32)
    segment_list: List[Dict[str, object]] = []
    for segment_index in range(int(num_segments)):
        t_min = float(edges[segment_index].item())
        t_max = float(edges[segment_index + 1].item())
        if segment_index == int(num_segments) - 1:
            segment_mask = strip_mask_box & (tangent_coord_box >= t_min) & (tangent_coord_box <= t_max)
        else:
            segment_mask = strip_mask_box & (tangent_coord_box >= t_min) & (tangent_coord_box < t_max)
        segment_list.append(
            {
                "segment_index": int(segment_index),
                "t_min": t_min,
                "t_max": t_max,
                "mask": segment_mask,
            }
        )
    return segment_list


def _crop_probability_box(prob_map: torch.Tensor, box: BoxDict, class_idx: int) -> torch.Tensor:
    """从全图概率图中裁出当前条带 bbox 内的某一类概率图。"""
    if prob_map.dim() != 4:
        raise ValueError("prob_map must have shape (B, K, H, W)")
    box_info = _parse_box(box)
    return prob_map[
        int(box_info["batch_idx"]),
        int(class_idx),
        int(box_info["y_min"]):int(box_info["y_max"]) + 1,
        int(box_info["x_min"]):int(box_info["x_max"]) + 1,
    ].to(torch.float32)


def select_best_prompt_in_strip_segment(
    response_box: torch.Tensor,
    segment_mask: torch.Tensor,
    normal_coord_box: torch.Tensor,
    box: BoxDict,
    side_name: str,
    segment_index: int,
    min_score: Optional[float] = None,
) -> Optional[Dict[str, int | float]]:
    """在单个切线段内，为指定一侧选一个最高响应点。

    输入：
        response_box: Tensor, shape = (h_box, w_box)
            当前侧的 ordered prototype 响应图。
        segment_mask: Tensor, shape = (h_box, w_box)
            当前切线段的 mask。
        normal_coord_box: Tensor, shape = (h_box, w_box)
            条带框内每个像素相对条带中心的法线坐标。
        box: dict
            当前条带框 box。
        side_name: str
            - `"a"`: A 侧
            - `"b"`: B 侧
        segment_index: int
            当前是第几段。
        min_score: Optional[float]
            若不为 `None`，低于该分数的点会被直接丢弃。

    输出：
        point_dict: Optional[dict]
            若成功选到点，则返回：
            {
                "batch_idx": int,
                "y": int,
                "x": int,
                "score": float,
                "segment_index": int,
            }
    """
    if side_name not in {"a", "b"}:
        raise ValueError("side_name must be either 'a' or 'b'")
    if response_box.shape != segment_mask.shape or response_box.shape != normal_coord_box.shape:
        raise ValueError("response_box, segment_mask, and normal_coord_box must share the same shape")

    # 这里先挖掉中心线附近一小条带，是为了避免正负点都压在边界本身；
    # SAM 更需要站在边界两侧的支持点，而不是继续重复边界像素。
    center_exclusion = float(DEFAULT_SEGMENT_PROMPT_CONFIG["center_exclusion_ratio"]) * float(box["strip_half_width"])
    if side_name == "a":
        side_mask = normal_coord_box <= -center_exclusion
    else:
        side_mask = normal_coord_box >= center_exclusion

    valid_mask = segment_mask & side_mask
    if not bool(valid_mask.any()):
        # 如果严格排除中心线后这一段没有点，就退一步允许整半带参与；
        # 这样做的前提是第三步条带框已经把两侧厚度限制住了，不会轻易跨到远处无关区域。
        if side_name == "a":
            valid_mask = segment_mask & (normal_coord_box <= 0.0)
        else:
            valid_mask = segment_mask & (normal_coord_box >= 0.0)
    if not bool(valid_mask.any()):
        return None

    # 这里每段只留一个点，不做 top-k；
    # 分散性由“切线分段”保证，而不是靠同一块局部区域里的 NMS 去勉强维持。
    masked_scores = response_box.clone()
    masked_scores[~valid_mask] = -1e8
    flat_index = int(torch.argmax(masked_scores.reshape(-1)).item())
    best_score = float(masked_scores.reshape(-1)[flat_index].item())
    if min_score is not None and best_score < float(min_score):
        return None

    w_box = int(response_box.size(1))
    y_local = flat_index // w_box
    x_local = flat_index % w_box
    box_info = _parse_box(box)
    return {
        "batch_idx": int(box_info["batch_idx"]),
        "y": int(box_info["y_min"] + y_local),
        "x": int(box_info["x_min"] + x_local),
        "score": best_score,
        "segment_index": int(segment_index),
    }


def _attach_prompt_roles(point_list: List[Dict[str, int | float]], role_for_a: str, role_for_b: str) -> List[Dict[str, int | float | str]]:
    """给最终 prompt 点补上对器官 A/B 的角色语义。"""
    output_points: List[Dict[str, int | float | str]] = []
    for point in point_list:
        new_point: Dict[str, int | float | str] = dict(point)
        new_point["role_for_a"] = role_for_a
        new_point["role_for_b"] = role_for_b
        output_points.append(new_point)
    return output_points


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
    """新版第四步主函数：在条带框内沿切线均匀分段取 prompts。

    输入：
        prob_map: Tensor, shape = (B, K, H, W)
            教师模型概率图。第一版默认不把它乘进响应，只作为可选加权输入保留。
        box: dict
            第三步输出的条带框 box。它必须已经带有：
            - `response_a_box`
            - `response_b_box`
            - `tangent_coord_box`
            - `normal_coord_box`
            - `strip_mask_box`
        core_ab/core_ba:
            为了兼容旧接口保留；在新版语义下，通常直接对应第三步得到的 `q_a / q_b`。
        a, b: int
            当前 pair 两侧类别 id。
        offset_distance/window_radius/sigma/min_distance:
            为了兼容旧接口保留；新版第四步不再使用这些旧含义。
        topk_per_side: int
            在新版里被解释成：沿切线方向平均切成多少段。
        min_score: Optional[float]
            每段最高响应若低于该阈值，则该段不输出 prompt。

    输出：
        result_dict: dict
            {
                "box": {...},
                "a": int,
                "b": int,
                "core_ab": {...},
                "core_ba": {...},
                "ref_center_a": {...},
                "ref_center_b": {...},
                "center_point": {...},
                "tangent_vector": Tensor[2],
                "normal_vector": Tensor[2],
                "points_a": [...],
                "points_b": [...],
                "segments": [...],
                "score_map_a_window": Tensor[h_box, w_box],
                "score_map_b_window": Tensor[h_box, w_box],
            }
    """
    if prob_map.dim() != 4:
        raise ValueError("prob_map must have shape (B, K, H, W)")

    box_info = _parse_box(box)
    strip_data = _extract_required_strip_data(box)
    response_a_box = strip_data["response_a_box"].clone()
    response_b_box = strip_data["response_b_box"].clone()
    tangent_coord_box = strip_data["tangent_coord_box"]
    normal_coord_box = strip_data["normal_coord_box"]
    strip_mask_box = strip_data["strip_mask_box"]

    if bool(DEFAULT_SEGMENT_PROMPT_CONFIG["use_probability_weight"]):
        # 第一版默认不乘概率，是为了先把“纯 ordered prototype 响应”链路看清楚；
        # 后面若要叠概率，也应该只是轻量加权，而不是重新引入硬门控。
        prob_a_box = _crop_probability_box(prob_map=prob_map, box=box, class_idx=int(a))
        prob_b_box = _crop_probability_box(prob_map=prob_map, box=box, class_idx=int(b))
        response_a_box = response_a_box * prob_a_box
        response_b_box = response_b_box * prob_b_box

    segments = split_strip_box_along_tangent(box=box, num_segments=max(1, int(topk_per_side)))

    raw_points_a: List[Dict[str, int | float]] = []
    raw_points_b: List[Dict[str, int | float]] = []
    for segment in segments:
        # 每段 A/B 各取一个点，这样输出数量天然和边界长度相关；
        # 比起在整个条带里一次性抢分，更容易得到沿边界均匀铺开的 prompts。
        best_a = select_best_prompt_in_strip_segment(
            response_box=response_a_box,
            segment_mask=segment["mask"],
            normal_coord_box=normal_coord_box,
            box=box,
            side_name="a",
            segment_index=int(segment["segment_index"]),
            min_score=min_score,
        )
        if best_a is not None:
            raw_points_a.append(best_a)

        best_b = select_best_prompt_in_strip_segment(
            response_box=response_b_box,
            segment_mask=segment["mask"],
            normal_coord_box=normal_coord_box,
            box=box,
            side_name="b",
            segment_index=int(segment["segment_index"]),
            min_score=min_score,
        )
        if best_b is not None:
            raw_points_b.append(best_b)

    points_a = _attach_prompt_roles(raw_points_a, role_for_a="positive", role_for_b="negative")
    points_b = _attach_prompt_roles(raw_points_b, role_for_a="negative", role_for_b="positive")

    center_point = box.get("center_point", {
        "batch_idx": int(box_info["batch_idx"]),
        "y": float(box.get("strip_center_y", 0.5 * (box_info["y_min"] + box_info["y_max"]))),
        "x": float(box.get("strip_center_x", 0.5 * (box_info["x_min"] + box_info["x_max"]))),
    })
    # 这里保留旧字段名 `ref_center_a / ref_center_b` 只是为了兼容外部入口；
    # 在新版语义下，它们更接近第三步的 `q_a / q_b`，而不是旧版的窗口中心。
    ref_center_a = dict(core_ab)
    ref_center_b = dict(core_ba)
    direction_vector = compute_local_boundary_direction(core_ab=core_ab, core_ba=core_ba)
    if direction_vector is None:
        direction_vector = torch.as_tensor(box["normal_vector"], dtype=torch.float32)

    return {
        "box": box,
        "a": int(a),
        "b": int(b),
        "pair": (int(a), int(b)),
        "core_ab": dict(core_ab),
        "core_ba": dict(core_ba),
        "ref_center_a": ref_center_a,
        "ref_center_b": ref_center_b,
        "center_point": center_point,
        "direction_vector": direction_vector,
        "tangent_vector": torch.as_tensor(box["tangent_vector"], dtype=torch.float32),
        "normal_vector": torch.as_tensor(box["normal_vector"], dtype=torch.float32),
        "q_a": dict(box.get("q_a", core_ab)),
        "q_b": dict(box.get("q_b", core_ba)),
        "boundary_points_global": box.get("boundary_points_global"),
        "normal_samples_a": box.get("normal_samples_a"),
        "normal_samples_b": box.get("normal_samples_b"),
        "similarity_curve_a": box.get("similarity_curve_a"),
        "similarity_curve_b": box.get("similarity_curve_b"),
        "smooth_similarity_curve_a": box.get("smooth_similarity_curve_a"),
        "smooth_similarity_curve_b": box.get("smooth_similarity_curve_b"),
        "changepoint_index_a": box.get("changepoint_index_a"),
        "changepoint_index_b": box.get("changepoint_index_b"),
        "segments": segments,
        "segment_regions": segments,
        "points_a": points_a,
        "points_b": points_b,

        # 这里继续输出 `window_a / window_b` 只是为了兼容老调用方；
        # 新版第四步实际只有一个条带框，不再有“两侧各自独立窗口”的几何对象。
        "window_a": box,
        "window_b": box,
        "score_map_a_window": response_a_box,
        "score_map_b_window": response_b_box,
        "response_a_box": response_a_box,
        "response_b_box": response_b_box,
        "tangent_coord_box": tangent_coord_box,
        "normal_coord_box": normal_coord_box,
        "strip_mask_box": strip_mask_box,
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
    """对多个条带框批量生成 point prompts。"""
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
    """可视化新版第四步的条带框分段 prompt 选择结果。

    这张调试图会尽量画出：
        1. 粗边界点
        2. 中心点 `c`
        3. 局部切线 `t`
        4. 局部法线 `n`
        5. 法线两侧采样点
        6. 变点 `q_a / q_b`
        7. 最终条带框与分段边界线
        8. 每一段选出的 positive / negative prompts
        9. 若 box 里带有一维相似度曲线，则右侧同时画出 A / B 曲线
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
    image_np = image_or_mask_2d.detach().cpu().numpy()
    use_label_cmap = not torch.is_floating_point(image_or_mask_2d)
    cmap = "nipy_spectral" if use_label_cmap else "gray"

    parsed_box = _parse_box(box)
    tangent_vector = torch.as_tensor(box.get("tangent_vector", tangent_vector if tangent_vector is not None else torch.tensor([0.0, 1.0])), dtype=torch.float32)
    normal_vector = torch.as_tensor(box.get("normal_vector", normal_vector if normal_vector is not None else torch.tensor([1.0, 0.0])), dtype=torch.float32)
    center_point = box.get("center_point", center_point if center_point is not None else {
        "batch_idx": int(parsed_box["batch_idx"]),
        "y": 0.5 * (parsed_box["y_min"] + parsed_box["y_max"]),
        "x": 0.5 * (parsed_box["x_min"] + parsed_box["x_max"]),
    })
    if boundary_points_global is None and isinstance(box.get("boundary_points_global"), torch.Tensor):
        boundary_points_global = box.get("boundary_points_global")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))
    axis_img, axis_curve = axes
    axis_img.imshow(image_np, cmap=cmap, interpolation="nearest")

    rough_rect = Rectangle(
        (parsed_box["x_min"], parsed_box["y_min"]),
        parsed_box["x_max"] - parsed_box["x_min"] + 1,
        parsed_box["y_max"] - parsed_box["y_min"] + 1,
        fill=False,
        edgecolor="yellow",
        linewidth=1.4,
        linestyle="--",
        alpha=0.8,
    )
    axis_img.add_patch(rough_rect)

    if boundary_points_global is not None and isinstance(boundary_points_global, torch.Tensor) and boundary_points_global.numel() > 0:
        axis_img.scatter(
            boundary_points_global[:, 1].detach().cpu().numpy(),
            boundary_points_global[:, 0].detach().cpu().numpy(),
            s=10,
            c="white",
            linewidths=0.0,
            alpha=0.55,
            label="coarse boundary",
        )

    center_y = float(center_point["y"])
    center_x = float(center_point["x"])
    axis_img.scatter([center_x], [center_y], s=80, c="magenta", edgecolors="black", linewidths=0.7, label="center c")

    tangent_scale = float(box.get("strip_half_length", max(parsed_box["x_max"] - parsed_box["x_min"], parsed_box["y_max"] - parsed_box["y_min"]) * 0.5))
    normal_scale = float(box.get("strip_half_width", 6.0))
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

    if isinstance(box.get("normal_samples_a"), dict):
        points_tensor = torch.as_tensor(box["normal_samples_a"]["points"], dtype=torch.float32)
        axis_img.scatter(points_tensor[:, 1].cpu().numpy(), points_tensor[:, 0].cpu().numpy(), s=12, c="orange", alpha=0.45, linewidths=0.0, label="A samples")
    if isinstance(box.get("normal_samples_b"), dict):
        points_tensor = torch.as_tensor(box["normal_samples_b"]["points"], dtype=torch.float32)
        axis_img.scatter(points_tensor[:, 1].cpu().numpy(), points_tensor[:, 0].cpu().numpy(), s=12, c="lime", alpha=0.45, linewidths=0.0, label="B samples")

    parsed_core_ab = _parse_point(core_ab)
    parsed_core_ba = _parse_point(core_ba)
    axis_img.scatter([float(parsed_core_ab["x"])], [float(parsed_core_ab["y"])], s=130, c="red", marker="*", edgecolors="white", linewidths=0.8, label=f"q_{a}")
    axis_img.scatter([float(parsed_core_ba["x"])], [float(parsed_core_ba["y"])], s=130, c="cyan", marker="*", edgecolors="black", linewidths=0.8, label=f"q_{b}")

    if "strip_polygon" in box:
        strip_polygon = torch.as_tensor(box["strip_polygon"], dtype=torch.float32)
        polygon = Polygon(strip_polygon[:, [1, 0]].detach().cpu().numpy(), closed=True, fill=False, edgecolor="yellow", linewidth=2.0, alpha=0.95)
        axis_img.add_patch(polygon)

    if isinstance(box.get("segment_regions"), list):
        segment_regions = box["segment_regions"]
    else:
        segment_regions = split_strip_box_along_tangent(box=box, num_segments=max(1, len(points_a), len(points_b), 1))

    for segment in segment_regions[:-1]:
        t_edge = float(segment["t_max"])
        center_vec = torch.tensor([float(box.get("strip_center_y", center_y)), float(box.get("strip_center_x", center_x))], dtype=torch.float32)
        tangent_vec = tangent_vector.to(torch.float32)
        normal_vec = normal_vector.to(torch.float32)
        half_width = float(box.get("strip_half_width", normal_scale))
        line_center = center_vec + t_edge * tangent_vec
        p1 = line_center - half_width * normal_vec
        p2 = line_center + half_width * normal_vec
        axis_img.plot([float(p1[1]), float(p2[1])], [float(p1[0]), float(p2[0])], color="yellow", linestyle="-.", linewidth=1.0, alpha=0.8)

    for point in points_a:
        parsed_point = _parse_point(point)
        segment_index = int(point.get("segment_index", 0)) if isinstance(point, dict) else 0
        axis_img.scatter([float(parsed_point["x"])], [float(parsed_point["y"])], s=82, c="orange", marker="o", edgecolors="black", linewidths=0.6, alpha=0.95)
        axis_img.text(float(parsed_point["x"]) + 0.8, float(parsed_point["y"]) + 0.8, f"A-S{segment_index + 1}", color="orange", fontsize=8, ha="left", va="bottom", bbox={"facecolor": "black", "alpha": 0.30, "pad": 1})
    for point in points_b:
        parsed_point = _parse_point(point)
        segment_index = int(point.get("segment_index", 0)) if isinstance(point, dict) else 0
        axis_img.scatter([float(parsed_point["x"])], [float(parsed_point["y"])], s=82, c="lime", marker="x", linewidths=2.0, alpha=0.95)
        axis_img.text(float(parsed_point["x"]) + 0.8, float(parsed_point["y"]) - 0.8, f"B-S{segment_index + 1}", color="lime", fontsize=8, ha="left", va="top", bbox={"facecolor": "black", "alpha": 0.30, "pad": 1})

    axis_img.set_title(f"pair=({a},{b}), segmented point prompts")
    axis_img.axis("off")
    axis_img.legend(loc="upper right", framealpha=0.9, fontsize=8)

    if box.get("similarity_curve_a") is not None and box.get("similarity_curve_b") is not None:
        distances_a = torch.as_tensor(box["normal_samples_a"]["distances"], dtype=torch.float32).cpu().numpy() if isinstance(box.get("normal_samples_a"), dict) else None
        distances_b = torch.as_tensor(box["normal_samples_b"]["distances"], dtype=torch.float32).cpu().numpy() if isinstance(box.get("normal_samples_b"), dict) else None
        curve_a = torch.as_tensor(box["similarity_curve_a"], dtype=torch.float32).cpu().numpy()
        curve_b = torch.as_tensor(box["similarity_curve_b"], dtype=torch.float32).cpu().numpy()
        smooth_a = torch.as_tensor(box.get("smooth_similarity_curve_a", box["similarity_curve_a"]), dtype=torch.float32).cpu().numpy()
        smooth_b = torch.as_tensor(box.get("smooth_similarity_curve_b", box["similarity_curve_b"]), dtype=torch.float32).cpu().numpy()
        idx_a = int(box.get("changepoint_index_a", 0))
        idx_b = int(box.get("changepoint_index_b", 0))
        if distances_a is None:
            distances_a = list(range(len(curve_a)))
        if distances_b is None:
            distances_b = list(range(len(curve_b)))
        axis_curve.plot(distances_a, curve_a, color="orange", alpha=0.35, linewidth=1.2, label="A raw")
        axis_curve.plot(distances_a, smooth_a, color="orange", alpha=0.95, linewidth=2.0, label="A smooth")
        axis_curve.axvline(distances_a[min(idx_a, len(distances_a) - 1)], color="red", linestyle="--", linewidth=1.4, label="A change")
        axis_curve.plot(distances_b, curve_b, color="limegreen", alpha=0.35, linewidth=1.2, label="B raw")
        axis_curve.plot(distances_b, smooth_b, color="limegreen", alpha=0.95, linewidth=2.0, label="B smooth")
        axis_curve.axvline(distances_b[min(idx_b, len(distances_b) - 1)], color="cyan", linestyle="--", linewidth=1.4, label="B change")
        axis_curve.set_title("similarity curves on the local normal")
        axis_curve.set_xlabel("distance from center c")
        axis_curve.set_ylabel("cosine similarity")
        axis_curve.grid(alpha=0.25)
        axis_curve.legend(loc="best", framealpha=0.9, fontsize=8)
    else:
        axis_curve.imshow(torch.as_tensor(box["response_a_box"], dtype=torch.float32).detach().cpu().numpy(), cmap="inferno", interpolation="nearest")
        axis_curve.set_title("A-side ordered response")
        axis_curve.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_demo_inputs() -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """构造一个最小 demo：先假定第三步已经给出条带框数据。"""
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

    y_min, y_max = 18, 54
    x_min, x_max = 20, 52
    response_a_box = prob_a[y_min:y_max + 1, x_min:x_max + 1]
    response_b_box = prob_b[y_min:y_max + 1, x_min:x_max + 1]
    tangent_coord_box = y_coords[y_min:y_max + 1, x_min:x_max + 1] - 36.0
    normal_coord_box = x_coords[y_min:y_max + 1, x_min:x_max + 1] - 36.0
    strip_mask_box = normal_coord_box.abs() <= 5.5

    box = {
        "batch_idx": 0,
        "a": 1,
        "b": 2,
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "box": (x_min, y_min, x_max, y_max),
        "strip_center_y": 36.0,
        "strip_center_x": 36.0,
        "strip_half_length": 18.0,
        "strip_half_width": 5.5,
        "tangent_vector": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "normal_vector": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "strip_polygon": torch.tensor([[18.0, 30.5], [54.0, 30.5], [54.0, 41.5], [18.0, 41.5]], dtype=torch.float32),
        "center_point": {"batch_idx": 0, "y": 36.0, "x": 36.0},
        "q_a": {"batch_idx": 0, "y": 36.0, "x": 31.0, "score": 0.9},
        "q_b": {"batch_idx": 0, "y": 36.0, "x": 41.0, "score": 0.9},
        "response_a_box": response_a_box,
        "response_b_box": response_b_box,
        "tangent_coord_box": tangent_coord_box,
        "normal_coord_box": normal_coord_box,
        "strip_mask_box": strip_mask_box,
    }
    return prob_map, torch.argmax(prob_map[0], dim=0), box


def run_demo(save_dir: str | Path) -> Dict[str, object]:
    """运行第四部分最小 demo。"""
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    prob_map, label_2d, box = _build_demo_inputs()
    result = generate_point_prompts_from_ordered_cores(
        prob_map=prob_map,
        box=box,
        core_ab=box["q_a"],
        core_ba=box["q_b"],
        a=1,
        b=2,
        topk_per_side=4,
    )
    visualize_point_prompts_in_box(
        image_or_mask_2d=label_2d,
        box=result["box"],
        core_ab=result["core_ab"],
        core_ba=result["core_ba"],
        ref_center_a=result["ref_center_a"],
        ref_center_b=result["ref_center_b"],
        points_a=result["points_a"],
        points_b=result["points_b"],
        a=result["a"],
        b=result["b"],
        save_path=save_dir / "ordered_boundary_segmented_prompt_demo.png",
        center_point=result["center_point"],
        normal_vector=result["normal_vector"],
        tangent_vector=result["tangent_vector"],
    )
    print(f"num_points_a={len(result['points_a'])}")
    print(f"num_points_b={len(result['points_b'])}")
    return result


def main() -> None:
    """模块直接运行时执行最小 demo。"""
    run_demo(Path(__file__).resolve().parent / "outputs")


if __name__ == "__main__":
    main()
