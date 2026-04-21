"""第三部分：目标域 box 内有序边界核心点搜索模块。

当前版本只做一件事：
给定一个已经构造好的局部 box，在 box 内分别搜索
1. 最像 `(A, B)` 有序边界原型的核心点 `z_{A,B}`
2. 最像 `(B, A)` 有序边界原型的核心点 `z_{B,A}`

注意：
1. 当前模块不再做全图 hard gating，也不再产出全图 ordered score map。
2. 当前模块只输出“核心边界点”，不直接生成最终给 SAM 用的 prompt。
3. `(A, B)` 与 `(B, A)` 必须严格分开处理，绝不合并成 unordered prototype。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from dynamic_boundary_prototype_bank import DynamicBoundaryPrototypeBank


BoundaryKey = Tuple[int, int]
BoundaryBox = Dict[str, int | Tuple[int, int, int, int]]
CorePoint = Dict[str, int | float | BoundaryKey]
PrototypeLibrary = (
    DynamicBoundaryPrototypeBank
    | Dict[BoundaryKey, Dict[str, torch.Tensor | int | float]]
    | Dict[BoundaryKey, torch.Tensor]
)


def canonicalize_ordered_boundary_key(a: int, b: int) -> BoundaryKey:
    """规范化有序边界 key。

    输入：
        a: int
            当前点所属类别。
        b: int
            当前点邻接的另一类。

    输出：
        boundary_key: Tuple[int, int]
            固定返回 `(a, b)`。

    说明：
        1. 这里绝不排序，也绝不合并。
        2. `(A, B)` 与 `(B, A)` 在当前模块里是两个不同的类别。
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
    box: BoundaryBox | Dict[str, int | Tuple[int, int, int, int]],
    batch_size: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Dict[str, int]:
    """把 box 统一解析成标准字典。

    支持的输入格式：
        1. {
               "batch_idx": int,
               "x1": int, "y1": int, "x2": int, "y2": int
           }
        2. {
               "batch_idx": int,
               "box": (x1, y1, x2, y2)
           }

    坐标约定：
        1. `x1, y1, x2, y2` 都是闭区间坐标。
        2. 返回后的 box 也继续使用闭区间表示。
    """
    if not isinstance(box, dict):
        raise TypeError("box must be a dictionary")

    batch_idx = int(box.get("batch_idx", 0))
    if "box" in box:
        raw_box = box["box"]
        if not isinstance(raw_box, (tuple, list)) or len(raw_box) != 4:
            raise ValueError("box['box'] must be a tuple/list with four elements: (x1, y1, x2, y2)")
        x1, y1, x2, y2 = [int(v) for v in raw_box]
    else:
        required_keys = ("x1", "y1", "x2", "y2")
        missing_keys = [key for key in required_keys if key not in box]
        if missing_keys:
            raise KeyError(f"box is missing required keys: {missing_keys}")
        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])

    if x2 < x1 or y2 < y1:
        raise ValueError("box must satisfy x2 >= x1 and y2 >= y1")
    if batch_size is not None and not (0 <= batch_idx < batch_size):
        raise IndexError("box batch_idx is out of range")
    if width is not None and not (0 <= x1 <= x2 < width):
        raise IndexError("box x coordinates are out of range")
    if height is not None and not (0 <= y1 <= y2 < height):
        raise IndexError("box y coordinates are out of range")

    return {
        "batch_idx": batch_idx,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def _get_box_size(box_info: Dict[str, int]) -> Tuple[int, int]:
    """返回 box 的 `(h_box, w_box)`。"""
    h_box = int(box_info["y2"] - box_info["y1"] + 1)
    w_box = int(box_info["x2"] - box_info["x1"] + 1)
    return h_box, w_box


def _extract_prototype_from_library(
    prototype_library: PrototypeLibrary,
    a: int,
    b: int,
) -> Optional[torch.Tensor]:
    """从 prototype library 中提取某个 ordered key 对应的 prototype。

    输入：
        prototype_library:
            支持：
            1. `DynamicBoundaryPrototypeBank`
            2. `{(A, B): Tensor[C]}`
            3. `{(A, B): {"prototype": Tensor[C], ...}}`
        a: int
            当前点所属类别。
        b: int
            当前点邻接类别。

    输出：
        prototype: Optional[Tensor], shape = (C,)
            若存在则返回当前 ordered key 对应的 prototype。
            若不存在则返回 `None`。
    """
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
    """在单个 box 内计算特征与某个 ordered prototype 的余弦相似度。

    输入：
        feature_map: Tensor, shape = (B, C, H, W)
            目标域特征图。
        box: dict
            局部框信息，支持：
            1. `{"batch_idx": b, "x1": x1, "y1": y1, "x2": x2, "y2": y2}`
            2. `{"batch_idx": b, "box": (x1, y1, x2, y2)}`
            坐标全部采用闭区间。
        prototype: Optional[Tensor], shape = (C,)
            某个 ordered boundary prototype。
            若为 `None`，表示当前 ordered key 缺少 prototype。
        normalize_feature: bool
            是否在当前函数内部对 box 内特征做 L2 normalize。
            若外部已经保证特征按通道归一化，可以传 `False`。
        missing_proto_value: Optional[float]
            prototype 缺失时的填充值。
            - `None`: 返回全零 map
            - 其它数值：返回该数值填充的 map

    输出：
        sim_map_box: Tensor, shape = (h_box, w_box)
            只覆盖当前 box 的局部相似度图。
            当前函数不会构造全图 score map。
    """
    if feature_map.dim() != 4:
        raise ValueError("feature_map must have shape (B, C, H, W)")

    batch_size, channels, height, width = feature_map.shape
    box_info = _parse_box(box, batch_size=batch_size, height=height, width=width)
    h_box, w_box = _get_box_size(box_info)

    if prototype is None:
        fill_value = 0.0 if missing_proto_value is None else float(missing_proto_value)
        return torch.full(
            (h_box, w_box),
            fill_value=fill_value,
            dtype=torch.float32,
            device=feature_map.device,
        )

    if prototype.dim() != 1 or prototype.numel() != channels:
        raise ValueError(f"prototype must have shape ({channels},)")

    batch_idx = box_info["batch_idx"]
    y1, y2 = box_info["y1"], box_info["y2"]
    x1, x2 = box_info["x1"], box_info["x2"]

    local_feature = feature_map[batch_idx, :, y1:y2 + 1, x1:x2 + 1].to(torch.float32)  # [C, h_box, w_box]
    local_feature = local_feature.permute(1, 2, 0).reshape(-1, channels)  # [h_box*w_box, C]
    if normalize_feature:
        local_feature = l2_normalize_feature(local_feature, dim=1)

    normalized_prototype = l2_normalize_feature(
        prototype.to(device=feature_map.device, dtype=torch.float32),
        dim=0,
    )  # prototype 这一步必须规范化，不然余弦相似度就不成立。

    similarity = torch.matmul(local_feature, normalized_prototype)  # [h_box*w_box]
    return similarity.reshape(h_box, w_box)


def _compute_soft_uncertainty_in_box(
    prob_map: torch.Tensor,
    box: BoundaryBox,
) -> torch.Tensor:
    """在 box 内计算软不确定性权重。

    输入：
        prob_map: Tensor, shape = (B, K, H, W)
            每个位置的类别概率图。
        box: dict
            与 `compute_ordered_similarity_in_box` 同格式的局部框。

    输出：
        uncertainty_box: Tensor, shape = (h_box, w_box)
            定义为：
            `u(x) = 1 - (top1_prob(x) - top2_prob(x))`

    说明：
        1. 这里只作为软权重，不做任何 hard gating。
        2. 输出会被 clamp 到 `[0, 1]`，避免出现无意义的负值。
    """
    if prob_map.dim() != 4:
        raise ValueError("prob_map must have shape (B, K, H, W)")

    batch_size, num_classes, height, width = prob_map.shape
    box_info = _parse_box(box, batch_size=batch_size, height=height, width=width)
    batch_idx = box_info["batch_idx"]
    y1, y2 = box_info["y1"], box_info["y2"]
    x1, x2 = box_info["x1"], box_info["x2"]

    local_prob = prob_map[batch_idx, :, y1:y2 + 1, x1:x2 + 1].to(torch.float32)  # [K, h_box, w_box]
    if num_classes == 1:
        return torch.ones_like(local_prob[0], dtype=torch.float32)

    top2_values = torch.topk(local_prob, k=2, dim=0, largest=True).values  # [2, h_box, w_box]
    uncertainty_box = 1.0 - (top2_values[0] - top2_values[1])
    return uncertainty_box.clamp_(min=0.0, max=1.0)


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
    """在单个 box 内分别计算 `(A,B)` 与 `(B,A)` 两个方向的有序响应。

    输入：
        feature_map: Tensor, shape = (B, C, H, W)
            目标域特征图。
        box: dict
            当前 pair 对应的局部框，格式见 `compute_ordered_similarity_in_box`。
        a: int
            第一个类别编号。
        b: int
            第二个类别编号。
        prototype_library:
            ordered prototype library。
        normalize_feature: bool
            是否在当前函数内部对 box 内特征做 L2 normalize。
        use_soft_uncertainty: bool
            是否启用软不确定性加权。
        prob_map: Optional[Tensor], shape = (B, K, H, W)
            当 `use_soft_uncertainty=True` 时，可提供类别概率图。

    输出：
        score_ab_box: Tensor, shape = (h_box, w_box)
            box 内 `(A, B)` 方向的响应图。
        score_ba_box: Tensor, shape = (h_box, w_box)
            box 内 `(B, A)` 方向的响应图。

    说明：
        1. 当前函数只在 box 内做搜索，不会生成 box 外的全图响应。
        2. 当前函数绝不会把 `(A, B)` 与 `(B, A)` 混成一个 unordered score。
    """
    prototype_ab = _extract_prototype_from_library(prototype_library, a=a, b=b)
    prototype_ba = _extract_prototype_from_library(prototype_library, a=b, b=a)

    score_ab_box = compute_ordered_similarity_in_box(
        feature_map=feature_map,
        box=box,
        prototype=prototype_ab,
        normalize_feature=normalize_feature,
        missing_proto_value=0.0,
    )
    score_ba_box = compute_ordered_similarity_in_box(
        feature_map=feature_map,
        box=box,
        prototype=prototype_ba,
        normalize_feature=normalize_feature,
        missing_proto_value=0.0,
    )

    if use_soft_uncertainty and prob_map is not None:
        uncertainty_box = _compute_soft_uncertainty_in_box(prob_map=prob_map, box=box)
        score_ab_box = uncertainty_box * score_ab_box
        score_ba_box = uncertainty_box * score_ba_box

    return score_ab_box, score_ba_box


def select_ordered_core_point_in_box(
    score_box: torch.Tensor,
    box: BoundaryBox,
    topk: int = 1,
    min_distance: float = 0.0,
) -> Optional[CorePoint | List[CorePoint]]:
    """从 box 内某个方向的局部 score 里选出核心点。

    输入：
        score_box: Tensor, shape = (h_box, w_box)
            单个方向的局部响应图。
        box: dict
            当前 box，格式见 `compute_ordered_similarity_in_box`。
        topk: int
            需要返回多少个候选点。
            - `topk=1` 时返回单个点字典或 `None`
            - `topk>1` 时返回点字典列表
        min_distance: float
            多点选择时的最小欧氏距离约束。
            若两个候选点距离过近，会跳过后面的点。

    输出：
        selected_core:
            - `topk=1`: `Optional[dict]`
            - `topk>1`: `List[dict]`

    说明：
        1. 当前函数输入的是 box 内局部 score。
        2. 返回时会统一转换成全图坐标。
        3. 如果 `score_box` 为空，或者没有任何有限值，则返回 `None` 或空列表。
    """
    if topk < 1:
        raise ValueError("topk must be >= 1")
    if score_box.dim() != 2:
        raise ValueError("score_box must have shape (h_box, w_box)")

    box_info = _parse_box(box)
    h_box, w_box = _get_box_size(box_info)
    if score_box.shape != (h_box, w_box):
        raise ValueError("score_box shape does not match the provided box size")
    if score_box.numel() == 0:
        return None if topk == 1 else []

    flat_score = score_box.reshape(-1)
    valid_mask = torch.isfinite(flat_score)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    if valid_indices.numel() == 0:
        return None if topk == 1 else []

    valid_scores = flat_score[valid_indices]
    sorted_local_order = torch.argsort(valid_scores, descending=True)
    sorted_flat_indices = valid_indices[sorted_local_order]

    selected_points: List[CorePoint] = []
    selected_local_coords: List[Tuple[int, int]] = []
    for flat_index in sorted_flat_indices.tolist():
        local_y = int(flat_index // w_box)
        local_x = int(flat_index % w_box)

        if min_distance > 0 and selected_local_coords:
            too_close = False
            for chosen_y, chosen_x in selected_local_coords:
                distance = ((local_y - chosen_y) ** 2 + (local_x - chosen_x) ** 2) ** 0.5
                if distance < float(min_distance):
                    too_close = True
                    break
            if too_close:
                continue

        global_y = int(box_info["y1"] + local_y)
        global_x = int(box_info["x1"] + local_x)
        selected_points.append(
            {
                "batch_idx": int(box_info["batch_idx"]),
                "y": global_y,
                "x": global_x,
                "score": float(score_box[local_y, local_x].item()),
            }
        )
        selected_local_coords.append((local_y, local_x))

        if len(selected_points) >= int(topk):
            break

    if topk == 1:
        return selected_points[0] if selected_points else None
    return selected_points


def generate_ordered_core_points_in_box(
    feature_map: torch.Tensor,
    box: BoundaryBox,
    a: int,
    b: int,
    prototype_library: PrototypeLibrary,
    prob_map: Optional[torch.Tensor] = None,
    use_soft_uncertainty: bool = False,
) -> Dict[str, object]:
    """对单个 pair-level box 生成双向有序核心点。

    输入：
        feature_map: Tensor, shape = (B, C, H, W)
            目标域特征图。
        box: dict
            当前 pair 对应的局部框。
        a: int
            第一个类别编号。
        b: int
            第二个类别编号。
        prototype_library:
            ordered prototype library。
        prob_map: Optional[Tensor], shape = (B, K, H, W)
            类别概率图，可选。
        use_soft_uncertainty: bool
            是否启用软不确定性加权。

    输出：
        result_dict: dict
            形式示例：
            {
                "box": {...},
                "pair": (A, B),
                "core_ab": {...} or None,
                "core_ba": {...} or None,
                "score_ab_box": Tensor[h_box, w_box],
                "score_ba_box": Tensor[h_box, w_box],
            }

    说明：
        1. `core_ab` 对应 `z_{A,B}`，表示“属于 A、邻接 B”的核心点。
        2. `core_ba` 对应 `z_{B,A}`，表示“属于 B、邻接 A”的核心点。
        3. 这里输出的是核心边界点，不是最终 prompt。
    """
    if feature_map.dim() != 4:
        raise ValueError("feature_map must have shape (B, C, H, W)")

    batch_size, _, height, width = feature_map.shape
    box_info = _parse_box(box, batch_size=batch_size, height=height, width=width)

    score_ab_box, score_ba_box = compute_ordered_boundary_core_scores_in_box(
        feature_map=feature_map,
        box=box_info,
        a=a,
        b=b,
        prototype_library=prototype_library,
        normalize_feature=True,
        use_soft_uncertainty=use_soft_uncertainty,
        prob_map=prob_map,
    )

    prototype_ab = _extract_prototype_from_library(prototype_library, a=a, b=b)
    prototype_ba = _extract_prototype_from_library(prototype_library, a=b, b=a)

    core_ab = None
    if prototype_ab is not None:
        selected_ab = select_ordered_core_point_in_box(score_box=score_ab_box, box=box_info, topk=1)
        if isinstance(selected_ab, dict):
            selected_ab["ordered_boundary_key"] = canonicalize_ordered_boundary_key(a, b)
            core_ab = selected_ab

    core_ba = None
    if prototype_ba is not None:
        selected_ba = select_ordered_core_point_in_box(score_box=score_ba_box, box=box_info, topk=1)
        if isinstance(selected_ba, dict):
            selected_ba["ordered_boundary_key"] = canonicalize_ordered_boundary_key(b, a)
            core_ba = selected_ba

    return {
        "box": box_info,
        "pair": (int(a), int(b)),
        "core_ab": core_ab,
        "core_ba": core_ba,
        "score_ab_box": score_ab_box,
        "score_ba_box": score_ba_box,
    }


def generate_ordered_core_points_for_boxes(
    feature_map: torch.Tensor,
    box_list: List[Dict[str, object]],
    prototype_library: PrototypeLibrary,
    prob_map: Optional[torch.Tensor] = None,
    use_soft_uncertainty: bool = False,
) -> List[Dict[str, object]]:
    """对一组 box 批量生成双向有序核心点。

    输入：
        feature_map: Tensor, shape = (B, C, H, W)
            目标域特征图。
        box_list: List[dict]
            每个元素至少包含：
            - `batch_idx`
            - `a`
            - `b`
            - box 坐标，支持：
              1. `x1, y1, x2, y2`
              2. `box=(x1, y1, x2, y2)`
        prototype_library:
            ordered prototype library。
        prob_map: Optional[Tensor], shape = (B, K, H, W)
            类别概率图，可选。
        use_soft_uncertainty: bool
            是否启用软不确定性加权。

    输出：
        result_list: List[dict]
            每个 box 对应一个结果字典，内部包含：
            - `core_ab`
            - `core_ba`
            - `score_ab_box`
            - `score_ba_box`
    """
    result_list: List[Dict[str, object]] = []
    for box_record in box_list:
        if "a" not in box_record or "b" not in box_record:
            raise KeyError("each box record must contain 'a' and 'b'")
        result = generate_ordered_core_points_in_box(
            feature_map=feature_map,
            box=box_record,
            a=int(box_record["a"]),
            b=int(box_record["b"]),
            prototype_library=prototype_library,
            prob_map=prob_map,
            use_soft_uncertainty=use_soft_uncertainty,
        )
        result_list.append(result)
    return result_list


def visualize_ordered_core_points_in_box(
    image_or_mask_2d: torch.Tensor,
    box: BoundaryBox,
    core_ab: Optional[CorePoint],
    core_ba: Optional[CorePoint],
    a: int,
    b: int,
    save_path: str | Path,
) -> None:
    """可视化单个 box 内的双向有序核心点。

    输入：
        image_or_mask_2d: Tensor, shape = (H, W)
            用于可视化的二维图像或标签图。
        box: dict
            当前 box。
        core_ab: Optional[dict]
            `z_{A,B}`，即 A->B 方向核心点。
        core_ba: Optional[dict]
            `z_{B,A}`，即 B->A 方向核心点。
        a: int
            pair 中第一个类别。
        b: int
            pair 中第二个类别。
        save_path: str | Path
            保存路径。
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
    box_info = _parse_box(box, height=height, width=width)
    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image_np = image_or_mask_2d.detach().cpu().numpy()
    use_label_cmap = not torch.is_floating_point(image_or_mask_2d)
    cmap = "nipy_spectral" if use_label_cmap else "gray"

    fig, axis = plt.subplots(1, 1, figsize=(6.5, 6.5))
    axis.imshow(image_np, cmap=cmap, interpolation="nearest")

    rect = Rectangle(
        (box_info["x1"], box_info["y1"]),
        width=box_info["x2"] - box_info["x1"] + 1,
        height=box_info["y2"] - box_info["y1"] + 1,
        fill=False,
        edgecolor="yellow",
        linewidth=2.0,
        linestyle="--",
    )
    axis.add_patch(rect)

    if core_ab is not None:
        axis.scatter(
            [int(core_ab["x"])],
            [int(core_ab["y"])],
            s=90,
            c="red",
            marker="o",
            edgecolors="white",
            linewidths=0.8,
            label=f"{a}->{b}",
        )

    if core_ba is not None:
        axis.scatter(
            [int(core_ba["x"])],
            [int(core_ba["y"])],
            s=100,
            c="cyan",
            marker="x",
            linewidths=2.0,
            label=f"{b}->{a}",
        )

    axis.set_title(f"pair=({a},{b}), ordered core points")
    axis.axis("off")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_all_ordered_boundary_scores(*args, **kwargs) -> Dict[BoundaryKey, torch.Tensor]:
    """旧版全图 ordered score 接口占位。

    当前第三部分已经改成“box 内核心点搜索”。
    如果还调用这个函数，说明外部流程仍停留在旧版设计上，需要同步改掉。
    """
    raise NotImplementedError(
        "The third-part module now only supports box-level ordered core point search. "
        "Use `generate_ordered_core_points_in_box` or `generate_ordered_core_points_for_boxes` instead."
    )


def extract_ordered_boundary_prompt_seeds(*args, **kwargs) -> Dict[BoundaryKey, List[CorePoint]]:
    """旧版 prompt seed 接口占位。

    当前文件只输出 ordered core points，不直接产出最终 prompt。
    prompt generation 会在后续阶段单独处理。
    """
    raise NotImplementedError(
        "This module no longer generates final prompt seeds. "
        "It only outputs ordered core boundary points inside each local box."
    )


def _build_demo_feature_map() -> Tuple[torch.Tensor, Dict[BoundaryKey, torch.Tensor], BoundaryBox, Tuple[int, int], Tuple[int, int]]:
    """构造最小可运行 demo 所需的 dummy 特征、prototype 和 box。"""
    batch_size, channels, height, width = 1, 4, 64, 64
    feature_map = torch.randn(batch_size, channels, height, width, dtype=torch.float32) * 0.05

    prototype_ab = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    prototype_ba = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    prototype_library = {
        canonicalize_ordered_boundary_key(1, 2): prototype_ab,
        canonicalize_ordered_boundary_key(2, 1): prototype_ba,
    }

    box: BoundaryBox = {
        "batch_idx": 0,
        "x1": 18,
        "y1": 18,
        "x2": 45,
        "y2": 45,
    }

    ab_point = (27, 25)
    ba_point = (35, 37)

    feature_map[0, :, ab_point[0], ab_point[1]] = torch.tensor([1.0, 0.1, 0.0, 0.0], dtype=torch.float32)
    feature_map[0, :, ba_point[0], ba_point[1]] = torch.tensor([0.1, 1.0, 0.0, 0.0], dtype=torch.float32)

    # 在 box 内再补几处次高响应点，避免 demo 过于理想化。
    feature_map[0, :, 29, 28] = torch.tensor([0.65, 0.30, 0.0, 0.0], dtype=torch.float32)
    feature_map[0, :, 33, 34] = torch.tensor([0.25, 0.72, 0.0, 0.0], dtype=torch.float32)
    return feature_map, prototype_library, box, ab_point, ba_point


def run_demo(save_dir: str | Path | None = None) -> Dict[str, object]:
    """运行最小可执行 demo。

    demo 内容：
        1. 构造 dummy `feature_map`
        2. 构造 dummy ordered `prototype_library`
        3. 构造一个 dummy box
        4. 在 box 内计算 `score_ab_box`、`score_ba_box`
        5. 选出 `core_ab`、`core_ba`
        6. 打印核心点位置与分数
        7. 保存一张可视化图
    """
    torch.manual_seed(7)
    feature_map, prototype_library, box, ab_gt, ba_gt = _build_demo_feature_map()

    result = generate_ordered_core_points_in_box(
        feature_map=feature_map,
        box=box,
        a=1,
        b=2,
        prototype_library=prototype_library,
        prob_map=None,
        use_soft_uncertainty=False,
    )

    score_ab_box = result["score_ab_box"]
    score_ba_box = result["score_ba_box"]
    core_ab = result["core_ab"]
    core_ba = result["core_ba"]

    print("=" * 80)
    print("Ordered Boundary Core Point Demo")
    print("=" * 80)
    print(f"box={result['box']}")
    print(f"score_ab_box shape={tuple(score_ab_box.shape)}")
    print(f"score_ba_box shape={tuple(score_ba_box.shape)}")
    print(f"expected demo A->B point={ab_gt}, predicted core_ab={core_ab}")
    print(f"expected demo B->A point={ba_gt}, predicted core_ba={core_ba}")

    if save_dir is None:
        save_dir = Path(__file__).resolve().parent / "outputs"
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "ordered_core_points_demo.png"

    dummy_canvas = torch.zeros((64, 64), dtype=torch.float32)
    dummy_canvas[box["y1"]:box["y2"] + 1, box["x1"]:box["x2"] + 1] = 0.15
    dummy_canvas[ab_gt[0], ab_gt[1]] = 0.65
    dummy_canvas[ba_gt[0], ba_gt[1]] = 0.95

    visualize_ordered_core_points_in_box(
        image_or_mask_2d=dummy_canvas,
        box=box,
        core_ab=core_ab if isinstance(core_ab, dict) else None,
        core_ba=core_ba if isinstance(core_ba, dict) else None,
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
