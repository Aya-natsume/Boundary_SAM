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
    original_dim = binary_mask.dim()  # 先记住原始维度，恢复输出形状时要用。
    if original_dim == 2:  # 单张二维图最简单，补上 batch 和 channel 两个维度。
        mask_4d = binary_mask.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:  # 三维输入默认解释成 (N, H, W)，补一个 channel 维度。
        mask_4d = binary_mask.unsqueeze(1)
    elif original_dim == 4:  # 四维输入要求 channel=1，不然这个函数的语义就乱了。
        if binary_mask.size(1) != 1:
            raise ValueError("binary_mask with 4 dims must have shape (N, 1, H, W)")
        mask_4d = binary_mask
    else:  # 维度不在支持范围内时直接报错，别让后面悄悄炸。
        raise ValueError("binary_mask must have shape (H, W), (1, H, W), (B, H, W), or (B, 1, H, W)")
    mask_4d = (mask_4d > 0).to(dtype=torch.float32)  # 二值 mask 统一压成 0/1 float，池化会更自然。
    return mask_4d, original_dim


def _restore_boundary_shape(boundary_mask: torch.Tensor, original_dim: int) -> torch.Tensor:
    """把 4D 边界张量恢复回更贴近输入的形状。"""
    if original_dim == 2:  # 单张二维图恢复成 (H, W)。
        return boundary_mask[0, 0]
    if original_dim == 3:  # 三维输入恢复成 (N, H, W)。
        return boundary_mask[:, 0]
    return boundary_mask  # 四维输入直接保留 (N, 1, H, W)。


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
    if kernel_size < 1 or kernel_size % 2 == 0:  # 这里明确要求正奇数核，省得中心定义变得暧昧。
        raise ValueError("kernel_size must be a positive odd integer")
    if mode not in {"inner", "gradient"}:  # 当前只支持两种模式，别随手写个别的字符串进来。
        raise ValueError("mode must be either 'inner' or 'gradient'")

    mask_4d, original_dim = _prepare_binary_mask(binary_mask)  # 先把输入整理成统一的 4D 形式。
    padding = kernel_size // 2  # 对称 padding 可以保持输出尺寸不变。

    dilated = F.max_pool2d(mask_4d, kernel_size=kernel_size, stride=1, padding=padding)  # 膨胀直接用 max pooling 就够快。
    eroded = 1.0 - F.max_pool2d(1.0 - mask_4d, kernel_size=kernel_size, stride=1, padding=padding)  # 二值腐蚀可以借补集技巧实现。

    if mode == "inner":  # 内边界更适合当前任务，因为我们只想在器官内部靠边缘的位置做邻接判定。
        boundary_4d = (mask_4d > 0.5) & (eroded < 0.5)
    else:  # gradient 模式会更厚一点，先支持好，后面你如果想切换不用重写函数。
        boundary_4d = (dilated - eroded) > 0.0

    return _restore_boundary_shape(boundary_4d, original_dim)  # 把形状恢复回更贴近输入的形式。


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
    one_hot = F.one_hot(seg_label.long(), num_classes=num_classes).permute(0, 3, 1, 2).to(torch.float32)  # 先把标签转成 one-hot，邻域计数才好做。
    kernel = torch.ones(
        num_classes,
        1,
        kernel_size,
        kernel_size,
        device=seg_label.device,
        dtype=torch.float32,
    )  # 这里每个类别各用一个全 1 卷积核，意思很直接。
    padding = kernel_size // 2  # 保持输出分辨率不变。
    local_counts = F.conv2d(one_hot, kernel, padding=padding, groups=num_classes)  # groups=num_classes 可以让每个类别独立计数。
    return local_counts


def _stack_boundary_coord_lists(
    boundary_coord_lists: MutableMapping[BoundaryKey, List[torch.Tensor]],
) -> BoundaryDict:
    """把按列表暂存的坐标整理成正式字典。"""
    boundary_dict: BoundaryDict = {}  # 最终结果统一放进这个字典里。
    for boundary_key, coord_list in boundary_coord_lists.items():  # 每个边界类别单独整理，逻辑会清楚很多。
        if len(coord_list) == 0:  # 没有坐标就跳过，别留空壳。
            continue
        coords = torch.cat(coord_list, dim=0)  # 同一类的多段坐标拼起来，得到一个完整张量。
        boundary_dict[boundary_key] = {
            "coords": coords,
            "raw_count": int(coords.size(0)),
        }  # 原始候选点数顺手记下来，后面统计会用到。
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
    if seg_label.dim() != 3:  # 这里只接受 (B, H, W)，别让奇怪形状混进来。
        raise ValueError("seg_label must have shape (B, H, W)")
    if boundary_kernel < 1 or boundary_kernel % 2 == 0:  # 边界带核大小要求正奇数，不然中心定义会变脏。
        raise ValueError("boundary_kernel must be a positive odd integer")
    if neighbor_kernel is None:  # 如果用户没单独指定邻域核，就默认和边界带核保持一致。
        neighbor_kernel = boundary_kernel
    if neighbor_kernel < 1 or neighbor_kernel % 2 == 0:  # 邻域核同样要求正奇数。
        raise ValueError("neighbor_kernel must be a positive odd integer")

    seg_label = seg_label.long()  # 标签必须是 long，后面 one-hot 和索引都更稳。
    batch_size, height, width = seg_label.shape  # 把基础尺寸拿出来，调试时看着也清楚。
    if num_classes <= 1:  # 没有前景类别时就别继续了，这种输入没有边界可提。
        raise ValueError("num_classes must be larger than 1")

    local_counts = _build_local_class_count_map(seg_label, num_classes=num_classes, kernel_size=neighbor_kernel)  # 邻域判定范围单独用 neighbor_kernel 控制，别再和边界带厚度绑死。
    local_counts_hwk = local_counts.permute(0, 2, 3, 1).contiguous()  # 改成 (B, H, W, K) 后，按坐标索引取邻域计数会更顺手。
    boundary_coord_lists: DefaultDict[BoundaryKey, List[torch.Tensor]] = defaultdict(list)  # 先按列表暂存，最后统一堆叠。

    for organ_class in range(num_classes):  # 这里现在显式包含背景 0，这样才能把 (0, A) 这一侧原型也建出来。
        organ_mask = seg_label == organ_class  # 当前类别的二值区域，可能是背景，也可能是前景器官。
        if not organ_mask.any():  # 这个类别在当前 batch 里根本没出现时，直接跳过，别浪费算力。
            continue

        candidate_boundary = extract_morph_boundary(organ_mask, kernel_size=boundary_kernel, mode="inner")  # 候选边界带厚度现在只由 boundary_kernel 控制。
        candidate_coords = torch.nonzero(candidate_boundary, as_tuple=False)  # 现在的坐标格式天然就是 [batch_idx, y, x]。
        if candidate_coords.numel() == 0:  # 当前器官没有候选边界时，继续下一个器官。
            continue

        neighborhood_counts = local_counts_hwk[
            candidate_coords[:, 0],
            candidate_coords[:, 1],
            candidate_coords[:, 2],
        ]  # 一次性取出所有候选点的邻域类别计数，别逐点循环慢慢抠。

        neighbor_counts = neighborhood_counts.clone()  # 后面要改写当前类别自身的计数，所以先复制一份，避免污染原图。
        neighbor_counts[:, organ_class] = 0.0  # 当前像素自身所在类别不算“邻接边界对象”，否则每个点都会偏向自己。

        foreground_neighbor_counts = neighbor_counts.clone()  # 前景优先逻辑要单独看一份前景计数。
        foreground_neighbor_counts[:, 0] = 0.0  # 背景先暂时屏蔽掉，等“是否接触任何前景”这个判断做完再考虑。

        max_foreground_count, best_foreground_class = foreground_neighbor_counts.max(dim=1)  # 找到每个候选点最可能邻接的前景类别。
        background_count = neighbor_counts[:, 0]  # 背景计数单独拿出来，作为后备归类。

        has_foreground_neighbor = max_foreground_count > 0  # 只要邻域里出现前景，就优先判成“当前侧 -> 某个前景”的有序边界。
        has_background_neighbor = background_count > 0  # 如果没有前景但有背景，就退化成“当前侧 -> 背景”的有序边界。
        valid_mask = has_foreground_neighbor | has_background_neighbor  # 两者都没有时，这个点就不构成有效边界类别。
        if not valid_mask.any():  # 没有有效点就继续下一个器官。
            continue

        assigned_neighbor_class = torch.where(
            has_foreground_neighbor,
            best_foreground_class,
            torch.zeros_like(best_foreground_class),
        )  # 先按“前景优先，否则背景”的规则给每个点分一个唯一邻接类别。

        # 一个关键点别漏了：
        # 1. 当 organ_class > 0 时，这里会产生 (A, 0)，也就是“器官侧接触背景”。
        # 2. 当 organ_class == 0 时，这里会产生 (0, A)，也就是“背景侧接触器官”。
        # 3. 这正是后续为器官-背景边界构造双向 ordered prototype 所必需的那一半信息。
        # 4. 虽然背景 mask 的 inner boundary 会在整张图边框附近也出现候选点，但那些位置通常没有前景邻居，
        #    所以会被 valid_mask 自动筛掉，不会莫名其妙污染 (0, A) 原型。

        valid_coords = candidate_coords[valid_mask]  # 只保留真正能归类的点。
        valid_neighbor_class = assigned_neighbor_class[valid_mask]  # 这些点对应的邻接类别。

        unique_neighbor_classes = torch.unique(valid_neighbor_class)  # 按目标边界类别分桶，后面写字典才干净。
        for neighbor_class in unique_neighbor_classes.tolist():  # 这里只在“类别级”循环，规模很小，别紧张。
            class_mask = valid_neighbor_class == neighbor_class  # 选出当前边界类别下的全部点。
            coords_this_key = valid_coords[class_mask]  # 当前 key 对应的边界点坐标。
            boundary_key = canonicalize_ordered_boundary_key(organ_class, neighbor_class)  # 当前点属于 organ_class，邻域接触 neighbor_class，所以必须保序写成 (A, B)。
            boundary_coord_lists[boundary_key].append(coords_this_key)  # 先放进列表，最后再统一拼接。

    return _stack_boundary_coord_lists(boundary_coord_lists)  # 最后把坐标列表整理成正式输出。


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
    if feature_map.dim() != 4:  # 特征图必须是标准四维格式，别用错了层。
        raise ValueError("feature_map must have shape (B, C, H, W)")
    if coords.dim() != 2 or coords.size(1) != 3:  # 坐标必须是 [N, 3]，否则没法按批次和空间位置索引。
        raise ValueError("coords must have shape (N, 3)")

    batch_size, channels, height, width = feature_map.shape  # 把特征图尺寸拿出来，便于做边界检查。
    if coords.numel() == 0:  # 空坐标直接返回空特征，别强行索引把自己绊倒。
        return feature_map.new_zeros((0, channels))

    coords = coords.long()  # 高级索引用 long，别拿别的 dtype 试图赌运气。
    batch_idx = coords[:, 0]  # 第一列是 batch 索引。
    y_idx = coords[:, 1]  # 第二列是 y 坐标。
    x_idx = coords[:, 2]  # 第三列是 x 坐标。

    if batch_idx.min() < 0 or batch_idx.max() >= batch_size:  # 先做索引安全检查，坏坐标早点炸比晚点炸好。
        raise IndexError("coords batch indices are out of range")
    if y_idx.min() < 0 or y_idx.max() >= height or x_idx.min() < 0 or x_idx.max() >= width:
        raise IndexError("coords spatial indices are out of range")

    point_features = feature_map[batch_idx, :, y_idx, x_idx]  # PyTorch 高级索引会一次性返回 [N, C]，不用逐点循环。
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
    if not (0.0 < keep_ratio <= 1.0):  # keep_ratio 语义必须明确，别传个奇怪值进来。
        raise ValueError("keep_ratio must be in the range (0, 1]")
    if min_points < 1:  # 最小点数至少得是 1，否则逻辑就不成立了。
        raise ValueError("min_points must be >= 1")

    filtered_boundary_dict: BoundaryDict = {}  # 净化后的结果统一放这里。

    for raw_key, raw_data in boundary_dict.items():  # 每个有序边界类别独立净化，(A,B) 与 (B,A) 绝不混合筛选。
        boundary_key = canonicalize_ordered_boundary_key(raw_key[0], raw_key[1])  # 保险起见，再走一次有序 key 规范化。
        coords = raw_data["coords"]  # 当前边界类别的候选点坐标。
        if not isinstance(coords, torch.Tensor):  # 这里做个基础类型检查，省得错误静悄悄地扩散。
            raise TypeError("boundary_dict['coords'] must be a torch.Tensor")

        raw_count = int(coords.size(0))  # 候选点总数先记下来，后面统计要用。
        point_features = gather_point_features(feature_map, coords)  # 先把所有候选点的特征一次性取出来。

        if raw_count == 0:  # 空类别就直接返回空结果，别做无意义的归一化。
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

        normalized_features = F.normalize(point_features, p=2, dim=1, eps=1e-12)  # 先把点特征做 L2 normalize，后面的点积就是真余弦相似度。
        prototype = normalized_features.mean(dim=0)  # 单原型就直接取均值，先求一个临时中心。
        prototype = F.normalize(prototype.unsqueeze(0), p=2, dim=1, eps=1e-12).squeeze(0)  # 均值向量也要再归一化，不然相似度尺度会漂。
        similarity = torch.matmul(normalized_features, prototype)  # 每个点和 prototype 的余弦相似度一次性算完。

        skip_filter = raw_count < min_points or keep_ratio >= 1.0  # 点太少时不筛，keep_ratio=1 时也没必要装作在筛。
        if skip_filter:  # 跳过筛选时就全部保留，但统计信息必须写清楚。
            keep_indices = torch.arange(raw_count, device=coords.device)  # 全部保留的索引就是自然顺序。
        else:
            keep_count = max(1, math.ceil(raw_count * keep_ratio))  # 用 ceil 保证保留比例不至于被向下截断得太狠。
            keep_indices = torch.topk(similarity, k=keep_count, largest=True).indices  # 只取相似度最高的一部分点。
            keep_indices = keep_indices[torch.argsort(keep_indices)]  # 把索引重新按原始顺序排一下，阅读和可视化会稳一点。

        kept_coords = coords[keep_indices]  # 净化后保留的坐标。
        kept_features = point_features[keep_indices]  # 与保留坐标对应的原始特征。
        kept_similarity = similarity[keep_indices]  # 保留点对应的相似度分数。
        kept_count = int(kept_coords.size(0))  # 最终保留点数。
        actual_keep_ratio = float(kept_count / raw_count) if raw_count > 0 else 0.0  # 实际保留比例也记下来，后面看结果更直观。

        filtered_boundary_dict[boundary_key] = {
            "coords": kept_coords,
            "features": kept_features,
            "raw_count": raw_count,
            "kept_count": kept_count,
            "similarity": kept_similarity,
            "prototype": prototype,
            "skip_filter": bool(skip_filter),
            "actual_keep_ratio": actual_keep_ratio,
        }  # 当前边界类别的净化结果写入总字典。

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
    if feature_map.dim() != 4:  # 特征图必须是 (B, C, H, W)。
        raise ValueError("feature_map must have shape (B, C, H, W)")
    if seg_label.dim() != 3:  # 标签图必须是 (B, H, W)。
        raise ValueError("seg_label must have shape (B, H, W)")
    if feature_map.size(0) != seg_label.size(0) or feature_map.size(2) != seg_label.size(1) or feature_map.size(3) != seg_label.size(2):
        raise ValueError("feature_map and seg_label must share the same batch and spatial size")

    raw_boundary_dict = assign_fine_boundary_labels(
        seg_label=seg_label,
        num_classes=num_classes,
        boundary_kernel=boundary_kernel,
        neighbor_kernel=neighbor_kernel,
    )  # 第一步和第二步：先做几何候选提取，再做细粒度边界类别赋值。

    filtered_boundary_dict = filter_boundary_points_by_feature_consistency(
        boundary_dict=raw_boundary_dict,
        feature_map=feature_map,
        keep_ratio=keep_ratio,
        min_points=min_points,
    )  # 第三步：在同类边界内部做轻量特征净化。

    summary_info: Dict[BoundaryKey, Dict[str, int | float | bool]] = {}  # 第四步：整理摘要统计，后续打印和日志都方便。
    all_keys = sorted(filtered_boundary_dict.keys())  # 统一按 key 排序，输出会稳定一点。
    for boundary_key in all_keys:  # 每个边界类别单独汇总。
        raw_data = raw_boundary_dict[boundary_key]  # 原始候选信息。
        filtered_data = filtered_boundary_dict[boundary_key]  # 净化后信息。
        raw_count = int(raw_data["raw_count"])  # 候选点数。
        kept_count = int(filtered_data["kept_count"])  # 保留点数。
        summary_info[boundary_key] = {
            "raw_count": raw_count,
            "kept_count": kept_count,
            "actual_keep_ratio": float(filtered_data["actual_keep_ratio"]),
            "skip_filter": bool(filtered_data["skip_filter"]),
        }  # 至少把题目要求的统计项都放齐，别省略。

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

    matplotlib.use("Agg")  # 用无界面后端保存图片，服务器环境也能跑。
    import matplotlib.pyplot as plt

    if seg_label_2d.dim() == 3:  # 如果输入带一个多余维度，就把它挤掉。
        if seg_label_2d.size(0) != 1:
            raise ValueError("seg_label_2d with 3 dims must have shape (1, H, W)")
        seg_label_2d = seg_label_2d[0]
    if seg_label_2d.dim() != 2:  # 这里只画单张二维图，别传 batch 进来。
        raise ValueError("seg_label_2d must have shape (H, W) or (1, H, W)")

    seg_np = seg_label_2d.detach().cpu().numpy()  # 画图前先搬到 CPU，再转 numpy。
    save_path = Path(save_path)  # 路径统一转成 Path，后面创建目录会方便一点。
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 输出目录不存在就创建，省得保存时报错。

    sorted_keys = sorted(boundary_dict.keys())  # 固定 key 顺序，颜色和图例会稳定。
    cmap = plt.get_cmap("tab20", max(1, len(sorted_keys)))  # 边界类别不多时用 tab20 基本够看。

    plt.figure(figsize=(8, 8))  # 图像尺寸适中一点，人工检查会舒服些。
    plt.imshow(seg_np, cmap="nipy_spectral", interpolation="nearest", alpha=0.85)  # 底图直接画标签，区域关系一眼能看见。

    legend_handles = []  # 图例句柄单独收集，后面统一加。
    legend_labels = []  # 图例名称也单独收集，避免重复。
    total_points = 0  # 顺手统计一下总点数，放标题里更直观。

    for color_index, boundary_key in enumerate(sorted_keys):  # 每个边界类别用一种颜色画点。
        data = boundary_dict[boundary_key]  # 取出当前边界类别的数据。
        coords = data["coords"]  # 坐标张量应该是 [N, 3]。
        if not isinstance(coords, torch.Tensor) or coords.numel() == 0:  # 没点就跳过，别画空散点。
            continue
        coords = coords.detach().cpu()  # 画图前搬到 CPU。
        batch_mask = coords[:, 0] == 0  # 当前函数只画 batch 0，方便单图人工检查。
        coords_2d = coords[batch_mask]  # 只取属于第一张图的点。
        if coords_2d.numel() == 0:  # 当前边界类别在 batch 0 上没点时就不画。
            continue

        ys = coords_2d[:, 1].numpy()  # y 坐标。
        xs = coords_2d[:, 2].numpy()  # x 坐标。
        scatter = plt.scatter(
            xs,
            ys,
            s=10,
            c=[cmap(color_index)],
            marker="o",
            linewidths=0.0,
            alpha=0.95,
        )  # 点稍微小一点，才不会把底图整个盖住。
        legend_handles.append(scatter)  # 当前类别的散点句柄加入图例列表。
        legend_labels.append(f"{_boundary_key_to_string(boundary_key)}: {coords_2d.size(0)}")  # 图例里把点数也带上，检查更方便。
        total_points += int(coords_2d.size(0))  # 总点数累加一下。

    plt.title(f"Fine Boundary Points (batch=0, total={total_points})")  # 标题里写清 batch 和总点数。
    plt.axis("off")  # 人工检查时不需要坐标轴刻度，去掉更干净。
    if legend_handles:  # 只有真的画了点才显示图例，别让空图例站地方。
        plt.legend(legend_handles, legend_labels, loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()  # 自动整理边距，避免图例或标题被裁掉。
    plt.savefig(save_path, dpi=200, bbox_inches="tight")  # 直接保存到磁盘。
    plt.close()  # 画完及时关图，别让内存一点点涨。


def _create_dummy_seg_label(height: int = 96, width: int = 96) -> torch.Tensor:
    """构造一个最小可运行的 dummy 标签图。

    输出：
        seg_label: Tensor, shape = (1, H, W)
            其中包含背景 0 和 3 个前景类别。
    """
    seg_label = torch.zeros((1, height, width), dtype=torch.long)  # 先全部置为背景，后面再往上画几个器官块。
    seg_label[0, 12:48, 10:34] = 1  # 类别 1：左侧矩形器官，和类别 2 有一段直接接触边界。
    seg_label[0, 18:58, 34:64] = 2  # 类别 2：中间矩形器官，和类别 1 相邻，同时也暴露在背景边界上。
    seg_label[0, 54:84, 16:42] = 3  # 类别 3：下方矩形器官，主要形成器官-背景边界。

    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")  # 再加一个小圆块，让形状别太死板。
    circle_mask = (yy - 70) ** 2 + (xx - 70) ** 2 <= 9 ** 2  # 这个圆属于类别 2，用来制造一点弯曲边界。
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
    torch.manual_seed(7)  # demo 固定随机种子，这样每次结果都稳定一点。
    batch_size, height, width = seg_label.shape  # 取出空间尺寸，后面生成位置编码要用。
    num_classes = int(seg_label.max().item()) + 1  # 从标签里推一个类别数，够 demo 用。

    class_prototypes = torch.randn(num_classes, channels, dtype=torch.float32)  # 给每个语义类别先分配一个基础特征中心。
    class_prototypes[0] = torch.zeros(channels, dtype=torch.float32)  # 背景特征中心设得简单一点，便于观察。
    class_prototypes = F.normalize(class_prototypes, dim=1) * 2.0  # 先归一化再放大一点，类别间分离会更明显。

    base_feature = class_prototypes[seg_label]  # 先按标签把每个像素映射到对应类别原型，形状是 (B, H, W, C)。
    base_feature = base_feature + 0.15 * torch.randn_like(base_feature)  # 加一点高斯噪声，让特征别显得过于理想化。

    yy = torch.linspace(-1.0, 1.0, steps=height).view(1, height, 1, 1)  # y 方向位置编码，给特征一点空间变化。
    xx = torch.linspace(-1.0, 1.0, steps=width).view(1, 1, width, 1)  # x 方向位置编码，同理。
    base_feature[..., 0:1] = base_feature[..., 0:1] + 0.30 * yy  # 第一维特征混入 y 位置信息。
    base_feature[..., 1:2] = base_feature[..., 1:2] + 0.30 * xx  # 第二维特征混入 x 位置信息。

    corruption_mask = torch.zeros((batch_size, height, width), dtype=torch.bool)  # 再人为制造一小块脏区域，让净化步骤更有东西可筛。
    corruption_mask[:, 28:42, 30:44] = True  # 这块区域刚好覆盖一部分 1-2 边界附近，适合观察净化效果。
    base_feature[corruption_mask] = torch.randn_like(base_feature[corruption_mask]) * 2.5  # 直接塞进更离谱的噪声特征，模拟几何上靠近但特征上不一致的点。

    feature_map = base_feature.permute(0, 3, 1, 2).contiguous()  # 最后转成标准的 (B, C, H, W)。
    return feature_map


def main() -> None:
    """最小可运行 demo。"""
    seg_label = _create_dummy_seg_label()  # 第一步：构造一个包含背景和 3 个前景类别的 dummy 标签图。
    feature_map = _create_dummy_feature_map(seg_label=seg_label, channels=8)  # 第二步：构造与标签对齐的 dummy 特征图。
    num_classes = 4  # 背景 0 + 前景 1/2/3，一共 4 类。

    raw_boundary_dict, filtered_boundary_dict, summary_info = build_source_fine_boundary_points(
        feature_map=feature_map,
        seg_label=seg_label,
        num_classes=num_classes,
        boundary_kernel=3,
        neighbor_kernel=3,
        keep_ratio=0.8,
        min_points=10,
    )  # 第三步：完整跑通“候选提取 -> 类别赋值 -> 特征净化”的总流程。

    output_dir = Path(__file__).resolve().parent / "outputs"  # 把 demo 输出放到项目内的 outputs 目录，查找会方便一点。
    output_dir.mkdir(parents=True, exist_ok=True)  # 输出目录不存在就创建。
    raw_vis_path = output_dir / "fine_boundary_points_raw.png"  # 原始候选点可视化路径。
    filtered_vis_path = output_dir / "fine_boundary_points_filtered.png"  # 净化后点可视化路径。

    visualize_fine_boundary_points(seg_label_2d=seg_label[0], boundary_dict=raw_boundary_dict, save_path=raw_vis_path)  # 先把原始候选点画出来。
    visualize_fine_boundary_points(seg_label_2d=seg_label[0], boundary_dict=filtered_boundary_dict, save_path=filtered_vis_path)  # 再把净化后的点画出来。

    print("=" * 80)  # 打印分隔线，让 demo 输出更好读。
    print("Source Fine Boundary Point Demo")  # demo 标题。
    print("=" * 80)
    for boundary_key in sorted(summary_info.keys()):  # 按 key 排序打印，每次输出都会稳定。
        stats = summary_info[boundary_key]  # 当前边界类别的摘要统计。
        print(
            f"boundary={_boundary_key_to_string(boundary_key):>8s} | "
            f"raw={int(stats['raw_count']):4d} | "
            f"kept={int(stats['kept_count']):4d} | "
            f"actual_keep_ratio={float(stats['actual_keep_ratio']):.3f} | "
            f"skip_filter={bool(stats['skip_filter'])}"
        )  # 题目要求打印筛选前后的点数，这里一并把保留比例和是否跳筛也写出来。
    print("-" * 80)
    print(f"raw visualization saved to:      {raw_vis_path}")  # 把原始图保存路径打印出来，方便你直接查看。
    print(f"filtered visualization saved to: {filtered_vis_path}")  # 把净化后图保存路径也打印出来。



if __name__ == "__main__":  # 让这个文件可以直接独立运行。
    main()
