"""使用 PICAI 预训练分割网络特征的细粒度边界点可视化脚本。

本脚本专门做可视化，不把实验入口塞进工具文件里。
整体流程严格参考 `project/Boundary/Boundary_v2_contrast_pic/train.py` 的前半部分：
1. 使用同款 PICAI 源域数据读取方式。
2. 使用同款 Encoder / Decoder 网络定义。
3. 使用同款 SE_source / SD 预训练权重读取方式。
4. 从 `SD(SE_source(x), Seg_D2=True)` 提取真实网络特征。
5. 调用 `source_fine_boundary_points.py` 中的工具函数划分细粒度边界点。
6. 输出几何空间与“每个边界类自己的特征净化结果”可视化。

说明：
1. 当前脚本故意不做目标域，不做 prototype library，不做 SAM。
2. 当前脚本重点回答的问题是：
   - 几何空间上边界点分对没分对。
   - 每个边界类在真实网络特征空间里，筛选后是不是更干净。
3. 当前不再输出那种“把所有边界类混在一起做一张全局 PCA”的图，那种图很容易把问题问偏。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path("/home/chenxu/Aya/project")  # 参考项目根目录，Boundary 包就在这里。
if str(PROJECT_ROOT) not in sys.path:  # 确保脚本独立运行时也能导入参考项目模块。
    sys.path.insert(0, str(PROJECT_ROOT))

from Boundary.model.Seg import Decoder, Encoder  # 直接使用参考训练脚本同款网络定义。
from Boundary.picai_dataset import PICAITrainDataset  # 直接使用参考训练脚本同款 PICAI 数据集类。

from dynamic_boundary_prototype_bank import (
    DynamicBoundaryPrototypeBank,
    compute_image_level_boundary_prototypes,
    update_boundary_prototype_bank_from_filtered_points,
)
from ordered_boundary_prompt_score import (
    canonicalize_ordered_boundary_key,
    compute_all_ordered_boundary_scores,
    extract_ordered_boundary_prompt_seeds,
    generate_ordered_core_points_for_boxes,
)
from ordered_boundary_point_prompt_generation import (
    generate_point_prompts_for_box_list,
)
from source_fine_boundary_points import (
    BoundaryDict,
    BoundaryKey,
    assign_fine_boundary_labels,
    build_source_fine_boundary_points,
    gather_point_features,
    visualize_fine_boundary_points,
)


def boundary_key_to_string(boundary_key: BoundaryKey) -> str:
    """把边界 key 转成便于显示的字符串。"""
    return f"({boundary_key[0]}, {boundary_key[1]})"


def normalize_image_for_display(image_2d: torch.Tensor) -> torch.Tensor:
    """把单张图像归一化到 0~1，便于显示。"""
    image_2d = image_2d.to(torch.float32)  # 显示前统一用 float，省得类型乱飞。
    image_min = float(image_2d.min())  # 最小值先记下来。
    image_max = float(image_2d.max())  # 最大值也记下来。
    if image_max <= image_min:  # 退化图像就返回全零，至少别炸。
        return torch.zeros_like(image_2d, dtype=torch.float32)
    return (image_2d - image_min) / (image_max - image_min)  # 标准 min-max 归一化够用了。


def get_boundary_colors(sorted_keys: List[BoundaryKey]) -> Dict[BoundaryKey, Tuple[float, float, float, float]]:
    """给每个边界类别分配一个稳定颜色。"""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20", max(1, len(sorted_keys)))  # PICAI 边界类别不多，tab20 足够用。
    return {boundary_key: cmap(index) for index, boundary_key in enumerate(sorted_keys)}  # 构建稳定的 key->color 映射。


def extract_batch0_coords(boundary_dict: BoundaryDict, boundary_key: BoundaryKey) -> torch.Tensor:
    """提取某个边界类别在 batch 0 上的二维坐标。"""
    coords = boundary_dict.get(boundary_key, {}).get("coords")  # 从字典里取出坐标张量。
    if not isinstance(coords, torch.Tensor) or coords.numel() == 0:  # 没坐标时直接返回空。
        return torch.zeros((0, 2), dtype=torch.long)
    coords = coords.detach().cpu()  # 可视化前统一放到 CPU。
    coords = coords[coords[:, 0] == 0]  # 当前脚本只画 batch 0。
    if coords.numel() == 0:  # batch 0 上没点就返回空。
        return torch.zeros((0, 2), dtype=torch.long)
    return coords[:, 1:]  # 只保留 y/x 两列。


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """移除 DataParallel 训练留下的 module. 前缀。"""
    cleaned_state_dict: Dict[str, torch.Tensor] = {}  # 新状态字典单独构建，逻辑更直观。
    for key, value in state_dict.items():  # 每个键单独处理，别耍花样。
        new_key = key.replace("module.", "")  # 参考训练脚本的写法，直接去掉 module.。
        cleaned_state_dict[new_key] = value
    return cleaned_state_dict


def load_reference_source_models(device: torch.device) -> Tuple[Encoder, Decoder]:
    """按参考训练脚本的方式加载源域预训练编码器与解码器。"""
    se_source = Encoder().to(device)  # 源域编码器和参考脚本保持一致。
    seg_decoder = Decoder(num_class=3).to(device)  # PICAI 一共 3 类，和参考脚本保持一致。

    se_source_dict = torch.load(
        "/home/chenxu/Aya/project/Boundary/save/picai/SE_source_picai_L2.pth",
        map_location=device,
    )  # 源域编码器权重路径直接对齐参考脚本。
    sd_dict = torch.load(
        "/home/chenxu/Aya/project/Boundary/save/picai/SD_picai_L2.pth",
        map_location=device,
    )  # 解码器权重路径同样直接对齐。

    se_source.load_state_dict(strip_module_prefix(se_source_dict), strict=True)  # 加载前去掉 module. 前缀。
    seg_decoder.load_state_dict(strip_module_prefix(sd_dict), strict=True)  # 解码器同理。

    se_source.eval()  # 可视化时不需要训练态，直接 eval 更稳。
    seg_decoder.eval()  # 解码器也切到 eval。
    return se_source, seg_decoder


def load_picai_source_slice(
    patient_index: int = 0,
    slice_index: Optional[int] = None,
    data_root: str | Path = "/home/chenxu/dataset/picai",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """按参考数据集类加载一张 PICAI 源域切片。

    输入：
        patient_index: int
            病例索引。
        slice_index: Optional[int]
            切片索引。
            - 当为 None 时，自动选择当前病例中前景面积最大的切片。
        data_root: str | Path
            PICAI 数据根目录。

    输出：
        image_batch: Tensor, shape = (1, 1, H, W)
            可直接送入网络的单张源域图像。
        seg_batch: Tensor, shape = (1, H, W)
            对应标签图。
        image_2d: Tensor, shape = (H, W)
            单张二维图像，用于可视化。
        patient_index: int
            实际使用的病例索引。
        slice_index: int
            实际使用的切片索引。
    """
    dataset = PICAITrainDataset(
        data_root=str(Path(data_root).expanduser().resolve() / "train"),
        modality="t2w",
        n_slices=1,
    )  # 直接使用参考项目的数据集类，而且和 train.py 一样是 n_slices=1。

    if slice_index is None:  # 没指定切片时就自动找一张信息量大的。
        seg_volume = dataset.data_file["seg"][patient_index]  # 当前病例的整套标签切片。
        foreground_area = (seg_volume > 0).reshape(seg_volume.shape[0], -1).sum(axis=1)  # 每张切片的前景面积先算出来。
        slice_index = int(foreground_area.argmax())  # 直接取前景面积最大的那张，通常边界也最丰富。

    global_index = patient_index * dataset.slice_nums + int(slice_index)  # 按参考数据集的索引规则算出全局样本索引。
    sample = dataset[global_index]  # 读出单个样本。

    image_slice = torch.from_numpy(sample["t2w"].astype("float32"))  # 原数据是 numpy，这里转成 float tensor。
    seg_slice = torch.from_numpy(sample["seg"].astype("int64"))  # 标签也转成 tensor。

    image_batch = image_slice.unsqueeze(0)  # [1, 1, H, W]，直接对齐网络输入格式。
    seg_batch = seg_slice.squeeze(0).unsqueeze(0)  # [1, H, W]，直接对齐边界工具函数输入格式。
    image_2d = image_batch[0, 0].clone()  # 单张二维图像单独保留给可视化用。
    return image_batch, seg_batch, image_2d, patient_index, int(slice_index)


def choose_slice_index_from_seg_volume(seg_volume, top_rank: int = 0) -> int:
    """从一个病例的分割体数据里选出前景面积较大的切片索引。

    输入：
        seg_volume: numpy.ndarray 或 Tensor, shape = (D, H, W) 或 (D, 1, H, W)
            当前病例的整套标签切片。
        top_rank: int
            按前景面积从大到小排序后的名次。
            - 0 表示前景面积最大的切片。
            - 1 表示第二大的切片。

    输出：
        slice_index: int
            选中的切片索引。
    """
    seg_tensor = torch.as_tensor(seg_volume).to(torch.int64)  # 先统一转成 int64 tensor，避免 uint16 在 CPU 上比较时报错。
    if seg_tensor.dim() == 4 and seg_tensor.size(1) == 1:  # 有些数据会带一个单通道维度，这里先挤掉。
        seg_tensor = seg_tensor[:, 0]
    if seg_tensor.dim() != 3:  # 当前函数只接受标准体数据，不接受别的奇怪形状。
        raise ValueError("seg_volume must have shape (D, H, W) or (D, 1, H, W)")

    flat_foreground = (seg_tensor > 0).reshape(seg_tensor.size(0), -1).sum(dim=1)  # 每张切片的前景面积先算出来。
    sorted_indices = torch.argsort(flat_foreground, descending=True)  # 按前景面积从大到小排序。
    top_rank = max(0, min(int(top_rank), int(sorted_indices.numel()) - 1))  # 越界时直接裁回合法范围，避免炸掉。
    return int(sorted_indices[top_rank].item())


def extract_pretrained_feature_map(
    image_batch: torch.Tensor,
    se_source: Encoder,
    seg_decoder: Decoder,
    device: torch.device,
) -> torch.Tensor:
    """使用真实预训练源域网络提取边界划分所需特征图。"""
    with torch.no_grad():  # 可视化只做前向，不需要梯度。
        image_batch = image_batch.to(device)  # 输入搬到模型所在设备。
        feature_map = seg_decoder(se_source(image_batch), Seg_D2=True)  # 完全对齐 train.py 里注释过的 source_features 写法。
    return feature_map.detach().cpu()  # 后续边界划分和可视化都放到 CPU 就够了。


def extract_pretrained_logits_and_feature_map(
    image_batch: torch.Tensor,
    se_source: Encoder,
    seg_decoder: Decoder,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """同时提取分割 logits 与边界特征图。

    输入：
        image_batch: Tensor, shape = (1, 1, H, W)
            单张 PICAI 切片。
        se_source: Encoder
            预训练源域编码器。
        seg_decoder: Decoder
            预训练分割解码器。
        device: torch.device
            模型运行设备。

    输出：
        seg_logits: Tensor, shape = (1, K, H, W)
            分割 logits。
        feature_map: Tensor, shape = (1, C, H, W)
            `Seg_D2=True` 输出的边界特征图。
    """
    with torch.no_grad():  # 可视化只做前向，不需要梯度。
        image_batch = image_batch.to(device)  # 输入先搬到模型设备。
        encoded = se_source(image_batch)  # 参考训练脚本，先过编码器。
        seg_logits = seg_decoder(encoded)  # 正常分割分支的 logits。
        feature_map = seg_decoder(encoded, Seg_D2=True)  # 边界特征分支。
    return seg_logits.detach().cpu(), feature_map.detach().cpu()


def build_boundary_mask_dict_from_boundary_dict(
    boundary_dict: BoundaryDict,
    batch_size: int,
    height: int,
    width: int,
) -> Dict[BoundaryKey, torch.Tensor]:
    """把 ordered boundary 坐标字典转成布尔 mask 字典。

    输入：
        boundary_dict: dict
            ordered boundary 字典，至少包含 `coords: Tensor[N, 3]`。
        batch_size: int
            批大小。
        height: int
            图像高度。
        width: int
            图像宽度。

    输出：
        boundary_mask_dict: dict
            形式为：
            {
                (A, B): Tensor[B, H, W],   # bool
                ...
            }
    """
    boundary_mask_dict: Dict[BoundaryKey, torch.Tensor] = {}
    for boundary_key, boundary_data in boundary_dict.items():  # 每个有序边界类别单独建一张 mask。
        coords = boundary_data.get("coords")
        if not isinstance(coords, torch.Tensor):
            raise TypeError(f"boundary_dict[{boundary_key}]['coords'] must be a torch.Tensor")
        mask = torch.zeros((batch_size, height, width), dtype=torch.bool)  # 当前有序边界类别的空间 mask。
        if coords.numel() > 0:
            coords = coords.long()
            mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True  # 直接按坐标回填成空间 mask。
        boundary_mask_dict[boundary_key] = mask
    return boundary_mask_dict


def build_pair_box_prompt_dict_from_score_dict(
    score_dict: Dict[BoundaryKey, torch.Tensor],
    candidate_mask_dict: Dict[BoundaryKey, torch.Tensor],
    box_quantile: float = 0.80,
    min_pixels: int = 4,
    padding: int = 6,
) -> Dict[Tuple[int, int], Dict[str, int | float | Tuple[int, int] | Tuple[BoundaryKey, ...]]]:
    """从 ordered score 图中提取 pair-level box prompt。

    设计选择：
        1. point prompt 保持 ordered，也就是 `(A,B)` 与 `(B,A)` 分开。
        2. box prompt 更适合作为一个局部 conflict 区域，因此这里把 `(A,B)` 与 `(B,A)` 两侧高响应区域合并成一个 pair-level box。
        3. 这个 box 只是为了把局部边界区域框出来，不承担左右方向语义。

    输入：
        score_dict: dict
            ordered score 字典：
            {
                (A, B): Tensor[B, H, W],
                ...
            }
        candidate_mask_dict: dict
            ordered candidate mask 字典，与 score_dict 对齐。
        box_quantile: float
            在当前 ordered score 的有效区域内，取高分分位数作为 box 区域阈值。
        min_pixels: int
            至少多少个高分像素才认为能稳定地产生一个 box。
        padding: int
            box 四周的额外外扩像素。

    输出：
        pair_box_prompt_dict: dict
            形式为：
            {
                (min_class, max_class): {
                    "box": (x1, y1, x2, y2),
                    "ordered_keys": ((A, B), (B, A)),
                    "num_pixels": int,
                    "score_threshold": float,
                }
            }
    """
    pair_region_dict: Dict[Tuple[int, int], torch.Tensor] = {}  # pair 级高分区域先临时累计在这里。
    pair_meta_dict: Dict[Tuple[int, int], Dict[str, object]] = {}  # box 的元信息也顺手记下来。

    for boundary_key, score_map in score_dict.items():  # 每个 ordered key 先单独提它的一侧高分区域。
        if boundary_key not in candidate_mask_dict:
            continue
        candidate_mask = candidate_mask_dict[boundary_key]
        if candidate_mask.shape != score_map.shape:
            raise ValueError(f"candidate mask shape for {boundary_key} does not match score map shape")

        valid_scores = score_map[candidate_mask]  # 只在这一侧候选区域内部看分数。
        if valid_scores.numel() == 0:
            continue
        threshold = float(torch.quantile(valid_scores.to(torch.float32), q=float(box_quantile)).item())  # 取高分位阈值，尽量让 box 落在更核心的边界区域。
        high_mask = candidate_mask & (score_map >= threshold)  # 当前有序边界的一侧高分区域。
        if int(high_mask.sum().item()) < int(min_pixels):
            top_count = min(int(min_pixels), int(valid_scores.numel()))  # 高分区域太少时，退化成 top-k。
            flat_valid_indices = torch.nonzero(candidate_mask.reshape(-1), as_tuple=False).squeeze(1)
            top_local_indices = torch.topk(valid_scores, k=top_count, largest=True).indices
            selected_flat_indices = flat_valid_indices[top_local_indices]
            high_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
            high_mask.view(-1)[selected_flat_indices] = True

        pair_key = tuple(sorted((int(boundary_key[0]), int(boundary_key[1]))))  # box 只服务于边界对区域，所以 pair 级 key 用无方向形式记录更方便。
        if pair_key not in pair_region_dict:
            pair_region_dict[pair_key] = high_mask.clone()
            pair_meta_dict[pair_key] = {
                "ordered_keys": [boundary_key],
                "score_thresholds": [threshold],
            }
        else:
            pair_region_dict[pair_key] = pair_region_dict[pair_key] | high_mask  # 两侧高分区域取并集，形成同一条边界对的局部 box。
            pair_meta_dict[pair_key]["ordered_keys"].append(boundary_key)
            pair_meta_dict[pair_key]["score_thresholds"].append(threshold)

    pair_box_prompt_dict: Dict[Tuple[int, int], Dict[str, int | float | Tuple[int, int] | Tuple[BoundaryKey, ...]]] = {}
    for pair_key, pair_mask in pair_region_dict.items():  # 每个 pair 级高分区域再转成 box。
        coords = torch.nonzero(pair_mask, as_tuple=False)
        if coords.numel() == 0:
            continue
        _, ys, xs = coords[:, 0], coords[:, 1], coords[:, 2]  # 当前脚本 batch=1，但这里还是按完整格式写，稳一点。
        height = int(pair_mask.size(1))
        width = int(pair_mask.size(2))
        x1 = max(0, int(xs.min().item()) - int(padding))
        y1 = max(0, int(ys.min().item()) - int(padding))
        x2 = min(width - 1, int(xs.max().item()) + int(padding))
        y2 = min(height - 1, int(ys.max().item()) + int(padding))
        pair_box_prompt_dict[pair_key] = {
            "box": (x1, y1, x2, y2),
            "ordered_keys": tuple(sorted(pair_meta_dict[pair_key]["ordered_keys"])),
            "num_pixels": int(coords.size(0)),
            "score_threshold": float(sum(pair_meta_dict[pair_key]["score_thresholds"]) / len(pair_meta_dict[pair_key]["score_thresholds"])),
        }
    return pair_box_prompt_dict


def encode_coords(coords: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """把 [batch, y, x] 坐标编码成一维整数，便于做集合匹配。"""
    coords = coords.long()  # 先保证是 long。
    return coords[:, 0] * height * width + coords[:, 1] * width + coords[:, 2]  # 只要 H/W 固定，这个编码就是唯一的。


def compute_class_pca_projection(
    raw_features: torch.Tensor,
    kept_features: torch.Tensor,
    dropped_features: torch.Tensor,
    prototype: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """在单个边界类别内部做 PCA 投影。

    输入：
        raw_features: Tensor, shape = (N_raw, C)
            当前边界类别全部候选点特征。
        kept_features: Tensor, shape = (N_kept, C)
            当前边界类别保留点特征。
        dropped_features: Tensor, shape = (N_drop, C)
            当前边界类别被过滤点特征。
        prototype: Tensor, shape = (C,)
            当前边界类别的原型。

    输出：
        raw_proj: Tensor, shape = (N_raw, 2)
            raw 点二维投影。
        kept_proj: Tensor, shape = (N_kept, 2)
            kept 点二维投影。
        dropped_proj: Tensor, shape = (N_drop, 2)
            dropped 点二维投影。
        prototype_proj: Tensor, shape = (1, 2)
            prototype 二维投影。
    """
    raw_features = raw_features.to(torch.float32)  # PCA 前统一用 float32。
    kept_features = kept_features.to(torch.float32)  # kept 特征同理。
    dropped_features = dropped_features.to(torch.float32)  # dropped 特征也同理。
    prototype = prototype.to(torch.float32).unsqueeze(0)  # prototype 也补成 batch 形式，后面投影方便。

    if raw_features.size(0) == 0:  # 空类别就直接返回空张量，别硬做 PCA。
        empty = torch.zeros((0, 2), dtype=torch.float32)
        return empty, empty, empty, torch.zeros((1, 2), dtype=torch.float32)

    mean = raw_features.mean(dim=0, keepdim=True)  # 单类 PCA 的中心只由当前类 raw 特征决定。
    raw_centered = raw_features - mean  # raw 去中心化。
    kept_centered = kept_features - mean if kept_features.numel() > 0 else kept_features.new_zeros((0, raw_features.size(1)))  # kept 用同一个中心化规则。
    dropped_centered = dropped_features - mean if dropped_features.numel() > 0 else dropped_features.new_zeros((0, raw_features.size(1)))  # dropped 也一样。
    prototype_centered = prototype - mean  # prototype 用同一个中心。

    feature_dim = raw_centered.size(1)  # 原始特征维度。
    rank = min(2, raw_centered.size(0), feature_dim)  # PCA 最多投 2 维，但样本太少时就别强撑。
    if rank == 0:  # 理论兜底分支，别让奇怪输入炸。
        empty = torch.zeros((0, 2), dtype=torch.float32)
        return empty, empty, empty, torch.zeros((1, 2), dtype=torch.float32)

    if rank == 1:  # 只有一维主成分时，第二维补零就行。
        basis = F.normalize(raw_centered[0:1].T, dim=0, eps=1e-12).T if raw_centered.size(0) > 0 else raw_centered.new_zeros((1, feature_dim))
    else:
        _, _, basis = torch.pca_lowrank(raw_centered, q=rank, center=False)  # 单类内部 PCA，问的问题才对。
        basis = basis[:, :rank].T  # 转成 [rank, C]，后面矩阵乘法更直接。

    raw_proj = torch.matmul(raw_centered, basis.T)  # raw 点投影。
    kept_proj = torch.matmul(kept_centered, basis.T) if kept_centered.numel() > 0 else kept_centered.new_zeros((0, rank))  # kept 点投影。
    dropped_proj = torch.matmul(dropped_centered, basis.T) if dropped_centered.numel() > 0 else dropped_centered.new_zeros((0, rank))  # dropped 点投影。
    prototype_proj = torch.matmul(prototype_centered, basis.T)  # prototype 投影。

    if rank == 1:  # 一维时统一补第二维，方便画二维图。
        raw_proj = torch.cat([raw_proj, raw_proj.new_zeros((raw_proj.size(0), 1))], dim=1)
        kept_proj = torch.cat([kept_proj, kept_proj.new_zeros((kept_proj.size(0), 1))], dim=1)
        dropped_proj = torch.cat([dropped_proj, dropped_proj.new_zeros((dropped_proj.size(0), 1))], dim=1)
        prototype_proj = torch.cat([prototype_proj, prototype_proj.new_zeros((prototype_proj.size(0), 1))], dim=1)

    return raw_proj[:, :2], kept_proj[:, :2], dropped_proj[:, :2], prototype_proj[:, :2]


def visualize_fine_boundary_dashboard(
    image_2d: torch.Tensor,
    seg_label_2d: torch.Tensor,
    raw_boundary_dict: BoundaryDict,
    filtered_boundary_dict: BoundaryDict,
    save_path: str | Path,
    max_boundary_panels: int = 8,
) -> None:
    """绘制几何空间上的综合可视化面板。"""
    import matplotlib

    matplotlib.use("Agg")  # 无界面环境也能画图保存。
    import matplotlib.pyplot as plt

    save_path = Path(save_path)  # 路径转 Path。
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 输出目录不存在就创建。

    image_np = normalize_image_for_display(image_2d).detach().cpu().numpy()  # 原图归一化到 0~1 更好看。
    seg_np = seg_label_2d.detach().cpu().numpy()  # 标签图搬到 CPU。
    sorted_keys = sorted(set(raw_boundary_dict.keys()) | set(filtered_boundary_dict.keys()))  # raw 和 filtered 的 key 取并集。
    colors = get_boundary_colors(sorted_keys)  # 每个边界类别一个稳定颜色。

    num_detail_panels = min(max_boundary_panels, len(sorted_keys))  # 单独边界类别面板数量做个上限。
    total_panels = 4 + num_detail_panels  # 4 个总览 + 若干小面板。
    num_cols = 3 if total_panels > 4 else 2  # 简单控制列数。
    num_rows = math.ceil(total_panels / num_cols)  # 行数按总面板数自动算。

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5.5 * num_cols, 5.0 * num_rows))  # 图稍微大一点，人工看得舒服。
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]  # 不管 axes 几维，都统一拉平。

    axes[0].imshow(image_np, cmap="gray", interpolation="nearest")  # 原始图像。
    axes[0].set_title("PICAI Source Image")
    axes[0].axis("off")

    axes[1].imshow(seg_np, cmap="nipy_spectral", interpolation="nearest")  # 对应标签图。
    axes[1].set_title("Source Label")
    axes[1].axis("off")

    axes[2].imshow(image_np, cmap="gray", interpolation="nearest")  # raw 边界点总览。
    raw_total = 0  # raw 总点数。
    for boundary_key in sorted_keys:
        coords_2d = extract_batch0_coords(raw_boundary_dict, boundary_key)  # 当前类别在 batch 0 上的二维坐标。
        if coords_2d.numel() == 0:
            continue
        axes[2].scatter(
            coords_2d[:, 1].numpy(),
            coords_2d[:, 0].numpy(),
            s=7,
            c=[colors[boundary_key]],
            linewidths=0.0,
            alpha=0.90,
        )  # raw 点画到原图上。
        raw_total += int(coords_2d.size(0))
    axes[2].set_title(f"Raw Boundary Points ({raw_total})")
    axes[2].axis("off")

    axes[3].imshow(image_np, cmap="gray", interpolation="nearest")  # filtered 边界点总览。
    kept_total = 0  # kept 总点数。
    for boundary_key in sorted_keys:
        coords_2d = extract_batch0_coords(filtered_boundary_dict, boundary_key)
        if coords_2d.numel() == 0:
            continue
        axes[3].scatter(
            coords_2d[:, 1].numpy(),
            coords_2d[:, 0].numpy(),
            s=7,
            c=[colors[boundary_key]],
            linewidths=0.0,
            alpha=0.95,
        )  # kept 点画到原图上。
        kept_total += int(coords_2d.size(0))
    axes[3].set_title(f"Filtered Boundary Points ({kept_total})")
    axes[3].axis("off")

    for panel_index, boundary_key in enumerate(sorted_keys[:num_detail_panels], start=4):  # 后面的面板逐类展示 raw/kept 对比。
        axis = axes[panel_index]
        axis.imshow(seg_np, cmap="gray", interpolation="nearest", alpha=0.45)  # 用浅灰标签图做底。
        raw_coords = extract_batch0_coords(raw_boundary_dict, boundary_key)
        kept_coords = extract_batch0_coords(filtered_boundary_dict, boundary_key)
        if raw_coords.numel() > 0:
            axis.scatter(
                raw_coords[:, 1].numpy(),
                raw_coords[:, 0].numpy(),
                s=9,
                c=[colors[boundary_key]],
                marker="o",
                linewidths=0.0,
                alpha=0.45,
                label=f"raw={raw_coords.size(0)}",
            )  # raw 点用淡一点的圆点。
        if kept_coords.numel() > 0:
            axis.scatter(
                kept_coords[:, 1].numpy(),
                kept_coords[:, 0].numpy(),
                s=12,
                c=[colors[boundary_key]],
                marker="x",
                linewidths=0.8,
                alpha=0.95,
                label=f"kept={kept_coords.size(0)}",
            )  # kept 点用叉号，更容易区分。
        axis.set_title(boundary_key_to_string(boundary_key))
        axis.axis("off")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.9)

    for axis in axes[total_panels:]:  # 多余空面板直接关掉，别留白。
        axis.axis("off")

    fig.tight_layout()  # 自动整理排版。
    fig.savefig(save_path, dpi=220, bbox_inches="tight")  # 保存几何空间综合图。
    plt.close(fig)  # 及时关图。


def visualize_boundary_feature_cleaning(
    raw_boundary_dict: BoundaryDict,
    filtered_boundary_dict: BoundaryDict,
    feature_map: torch.Tensor,
    save_path: str | Path,
) -> None:
    """按边界类别可视化特征空间净化效果。

    每个边界类别画两张图：
    1. 当前边界类别内部的 PCA 二维投影，区分 raw / kept / dropped。
    2. 当前边界类别内部的 prototype 相似度分布，区分 kept / dropped。

    这才是在回答“这个边界类筛选后是否更干净”，而不是去看一个全局混合二维图。
    """
    import matplotlib

    matplotlib.use("Agg")  # 无界面环境一样能保存图片。
    import matplotlib.pyplot as plt

    save_path = Path(save_path)  # 路径转 Path。
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 目录不存在就创建。

    sorted_keys = sorted(raw_boundary_dict.keys())  # 只看原始存在的边界类别。
    if len(sorted_keys) == 0:  # 没边界点时直接返回，别硬画空图。
        return

    num_rows = len(sorted_keys)  # 每个边界类别占一行。
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 5 * num_rows))  # 左边 PCA，右边 similarity 直方图。
    if num_rows == 1:  # 单行时把 axes 形状统一成二维列表，后面代码会更干净。
        axes = [axes]

    for row_index, boundary_key in enumerate(sorted_keys):  # 每个边界类别单独分析，这样问的问题才对。
        raw_data = raw_boundary_dict[boundary_key]  # 当前边界类 raw 数据。
        filtered_data = filtered_boundary_dict[boundary_key]  # 当前边界类 filtered 数据。

        raw_coords = raw_data["coords"]  # 当前类全部 raw 点坐标。
        kept_coords = filtered_data["coords"]  # 当前类保留点坐标。
        raw_features = gather_point_features(feature_map, raw_coords)  # 当前类全部 raw 点特征。
        normalized_features = F.normalize(raw_features, p=2, dim=1, eps=1e-12)  # 和工具函数一样，先做 L2 normalize。
        prototype = filtered_data["prototype"]  # 直接使用工具函数输出的 prototype，别自己另起炉灶。
        raw_similarity = torch.matmul(normalized_features, prototype.to(normalized_features.device))  # 计算当前类 raw 点与 prototype 的相似度。

        height = feature_map.size(2)  # 编码坐标时要用 H。
        width = feature_map.size(3)  # 编码坐标时要用 W。
        raw_code = encode_coords(raw_coords, height=height, width=width)  # raw 坐标编码。
        kept_code = encode_coords(kept_coords, height=height, width=width) if kept_coords.numel() > 0 else raw_code.new_zeros((0,))  # kept 坐标编码。
        keep_mask = torch.isin(raw_code, kept_code) if kept_code.numel() > 0 else torch.zeros_like(raw_code, dtype=torch.bool)  # 用编码后的坐标做集合匹配，得到哪些点被保留。
        drop_mask = ~keep_mask  # 剩下的就是被过滤掉的点。

        kept_features = raw_features[keep_mask]  # 保留点特征。
        dropped_features = raw_features[drop_mask]  # 被过滤点特征。
        kept_similarity = raw_similarity[keep_mask]  # 保留点相似度。
        dropped_similarity = raw_similarity[drop_mask]  # 被过滤点相似度。

        raw_proj, kept_proj, dropped_proj, prototype_proj = compute_class_pca_projection(
            raw_features=raw_features,
            kept_features=kept_features,
            dropped_features=dropped_features,
            prototype=prototype,
        )  # 只在当前类内部做 PCA，避免不同类混投到一起误导判断。

        axis_pca = axes[row_index][0]  # 左图：类内特征空间投影。
        axis_hist = axes[row_index][1]  # 右图：类内相似度分布。

        if raw_proj.numel() > 0:
            axis_pca.scatter(
                raw_proj[:, 0].cpu().numpy(),
                raw_proj[:, 1].cpu().numpy(),
                s=12,
                c="lightgray",
                linewidths=0.0,
                alpha=0.45,
                label=f"raw={raw_proj.size(0)}",
            )  # 全部 raw 点先画成浅灰底层。
        if dropped_proj.numel() > 0:
            axis_pca.scatter(
                dropped_proj[:, 0].cpu().numpy(),
                dropped_proj[:, 1].cpu().numpy(),
                s=18,
                c="tab:red",
                linewidths=0.0,
                alpha=0.80,
                label=f"dropped={dropped_proj.size(0)}",
            )  # 被过滤点单独用红色标出来，看它们是不是偏离主簇。
        if kept_proj.numel() > 0:
            axis_pca.scatter(
                kept_proj[:, 0].cpu().numpy(),
                kept_proj[:, 1].cpu().numpy(),
                s=18,
                c="tab:blue",
                linewidths=0.0,
                alpha=0.85,
                label=f"kept={kept_proj.size(0)}",
            )  # 保留点用蓝色，便于观察筛选后主簇是否更紧。
        axis_pca.scatter(
            prototype_proj[:, 0].cpu().numpy(),
            prototype_proj[:, 1].cpu().numpy(),
            s=120,
            c="gold",
            marker="*",
            edgecolors="black",
            linewidths=0.7,
            label="prototype",
        )  # prototype 用星号单独标出来，方便看 kept / dropped 相对中心的关系。
        axis_pca.set_title(f"{boundary_key_to_string(boundary_key)} Feature PCA")  # 标题写清当前边界类。
        axis_pca.set_xlabel("PC-1")  # 第一主成分。
        axis_pca.set_ylabel("PC-2")  # 第二主成分。
        axis_pca.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)  # 加一点浅网格，肉眼看形状更方便。
        axis_pca.legend(loc="best", fontsize=8, framealpha=0.9)  # 图例说明 raw/kept/dropped/prototype。

        if raw_similarity.numel() > 0:
            axis_hist.hist(
                raw_similarity.cpu().numpy(),
                bins=24,
                color="lightgray",
                alpha=0.65,
                label=f"raw={raw_similarity.numel()}",
            )  # 全部 raw 相似度分布先画一层浅灰底图。
        if dropped_similarity.numel() > 0:
            axis_hist.hist(
                dropped_similarity.cpu().numpy(),
                bins=24,
                color="tab:red",
                alpha=0.65,
                label=f"dropped={dropped_similarity.numel()}",
            )  # 被过滤点的相似度分布。
        if kept_similarity.numel() > 0:
            axis_hist.hist(
                kept_similarity.cpu().numpy(),
                bins=24,
                color="tab:blue",
                alpha=0.65,
                label=f"kept={kept_similarity.numel()}",
            )  # 保留点的相似度分布。
            if not bool(filtered_data["skip_filter"]):  # 真做了筛选时，再把切分阈值画出来。
                threshold = float(kept_similarity.min().item())  # keep 集合里的最小相似度就是当前截断点。
                axis_hist.axvline(threshold, color="black", linestyle="--", linewidth=1.0, label=f"cutoff={threshold:.3f}")  # 把 cut-off 画出来，更直观。
        axis_hist.set_title(f"{boundary_key_to_string(boundary_key)} Similarity Histogram")  # 标题说明当前边界类。
        axis_hist.set_xlabel("Cosine Similarity to Prototype")  # 横轴就是当前方法真正用来筛选的量。
        axis_hist.set_ylabel("Count")  # 纵轴是点数。
        axis_hist.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)  # 继续加浅网格。
        axis_hist.legend(loc="best", fontsize=8, framealpha=0.9)  # 图例说明分布。

    fig.tight_layout()  # 整理排版，别让图例和标题互相打架。
    fig.savefig(save_path, dpi=220, bbox_inches="tight")  # 保存特征净化可视化结果。
    plt.close(fig)  # 及时关图，别让 figure 挂着吃内存。


def compute_class_pca_projection_with_bank(
    raw_features: torch.Tensor,
    kept_features: torch.Tensor,
    dropped_features: torch.Tensor,
    image_proto: torch.Tensor,
    bank_proto: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """在单个边界类别内部做 PCA 投影，并把图级与 bank prototype 一起投进去。"""
    raw_features = raw_features.to(torch.float32)  # PCA 前统一转 float32。
    kept_features = kept_features.to(torch.float32)  # kept 特征也统一一下。
    dropped_features = dropped_features.to(torch.float32)  # dropped 特征也统一一下。
    image_proto = image_proto.to(torch.float32).unsqueeze(0)  # 图级 prototype 补成 [1, C]。
    bank_proto_tensor = None if bank_proto is None else bank_proto.to(torch.float32).unsqueeze(0)  # bank prototype 可能不存在，所以先做可选处理。

    if raw_features.size(0) == 0:  # 空类别时直接返回空结果，别硬做 PCA。
        empty = torch.zeros((0, 2), dtype=torch.float32)
        image_proj = torch.zeros((1, 2), dtype=torch.float32)
        bank_proj = None if bank_proto_tensor is None else torch.zeros((1, 2), dtype=torch.float32)
        return empty, empty, empty, image_proj, bank_proj

    mean = raw_features.mean(dim=0, keepdim=True)  # 当前类的中心只由当前类 raw 特征决定。
    raw_centered = raw_features - mean  # raw 去中心化。
    kept_centered = kept_features - mean if kept_features.numel() > 0 else kept_features.new_zeros((0, raw_features.size(1)))  # kept 用同一中心化规则。
    dropped_centered = dropped_features - mean if dropped_features.numel() > 0 else dropped_features.new_zeros((0, raw_features.size(1)))  # dropped 同理。
    image_proto_centered = image_proto - mean  # 图级 prototype 也放到同一个坐标系里。
    bank_proto_centered = None if bank_proto_tensor is None else bank_proto_tensor - mean  # bank prototype 同理。

    feature_dim = raw_centered.size(1)  # 原始特征维度。
    rank = min(2, raw_centered.size(0), feature_dim)  # PCA 最多投 2 维，但样本太少时别强撑。
    if rank == 0:  # 理论兜底分支。
        empty = torch.zeros((0, 2), dtype=torch.float32)
        image_proj = torch.zeros((1, 2), dtype=torch.float32)
        bank_proj = None if bank_proto_centered is None else torch.zeros((1, 2), dtype=torch.float32)
        return empty, empty, empty, image_proj, bank_proj

    if rank == 1:  # 只有一维主成分时，第二维补零就行。
        basis = F.normalize(raw_centered[0:1].T, dim=0, eps=1e-12).T if raw_centered.size(0) > 0 else raw_centered.new_zeros((1, feature_dim))
    else:
        _, _, basis = torch.pca_lowrank(raw_centered, q=rank, center=False)  # 只在当前类内部做 PCA，这样问的问题才对。
        basis = basis[:, :rank].T  # 转成 [rank, C]，矩阵乘法更直接。

    raw_proj = torch.matmul(raw_centered, basis.T)  # raw 点投影。
    kept_proj = torch.matmul(kept_centered, basis.T) if kept_centered.numel() > 0 else kept_centered.new_zeros((0, rank))  # kept 点投影。
    dropped_proj = torch.matmul(dropped_centered, basis.T) if dropped_centered.numel() > 0 else dropped_centered.new_zeros((0, rank))  # dropped 点投影。
    image_proj = torch.matmul(image_proto_centered, basis.T)  # 图级 prototype 投影。
    bank_proj = None if bank_proto_centered is None else torch.matmul(bank_proto_centered, basis.T)  # bank prototype 投影。

    if rank == 1:  # 一维时统一补第二维，方便画二维图。
        raw_proj = torch.cat([raw_proj, raw_proj.new_zeros((raw_proj.size(0), 1))], dim=1)
        kept_proj = torch.cat([kept_proj, kept_proj.new_zeros((kept_proj.size(0), 1))], dim=1)
        dropped_proj = torch.cat([dropped_proj, dropped_proj.new_zeros((dropped_proj.size(0), 1))], dim=1)
        image_proj = torch.cat([image_proj, image_proj.new_zeros((image_proj.size(0), 1))], dim=1)
        if bank_proj is not None:
            bank_proj = torch.cat([bank_proj, bank_proj.new_zeros((bank_proj.size(0), 1))], dim=1)

    return raw_proj[:, :2], kept_proj[:, :2], dropped_proj[:, :2], image_proj[:, :2], None if bank_proj is None else bank_proj[:, :2]


def build_boundary_prototype_bank_from_reference_subset(
    device: torch.device,
    se_source: Encoder,
    seg_decoder: Decoder,
    patient_indices: Tuple[int, ...],
    slice_ranks: Tuple[int, ...],
    boundary_kernel: int,
    neighbor_kernel: int,
    keep_ratio: float,
    min_points: int,
) -> DynamicBoundaryPrototypeBank:
    """从少量真实 PICAI 源域切片构建动态 EMA bank。"""
    bank: Optional[DynamicBoundaryPrototypeBank] = None  # bank 先留空，等第一次拿到特征维度再初始化。

    for patient_index in patient_indices:  # 先在病例级循环，规模不大，完全可控。
        for slice_rank in slice_ranks:  # 每个病例只取少量前景面积较大的切片，别把整套数据都扫了。
            image_batch, seg_batch, _, _, used_slice_index = load_picai_source_slice(
                patient_index=int(patient_index),
                slice_index=None,
            ) if int(slice_rank) == 0 else load_picai_source_slice(
                patient_index=int(patient_index),
                slice_index=choose_slice_index_from_seg_volume(
                    PICAITrainDataset(
                        data_root=str(Path("/home/chenxu/dataset/picai").expanduser().resolve() / "train"),
                        modality="t2w",
                        n_slices=1,
                    ).data_file["seg"][int(patient_index)],
                    top_rank=int(slice_rank),
                ),
            )  # 当前写法有点直白，但目的是只取少量真实切片做 bank，可视化够用了。

            feature_map = extract_pretrained_feature_map(
                image_batch=image_batch,
                se_source=se_source,
                seg_decoder=seg_decoder,
                device=device,
            )  # 用真实预训练网络提特征。

            if bank is None:  # 第一次拿到特征图时再初始化 bank，这样 feature_dim 不会猜错。
                bank = DynamicBoundaryPrototypeBank(
                    feature_dim=int(feature_map.size(1)),
                    momentum=0.9,
                    device="cpu",
                )  # 当前可视化 bank 放 CPU 就够了。

            _, filtered_boundary_dict, _ = build_source_fine_boundary_points(
                feature_map=feature_map,
                seg_label=seg_batch,
                num_classes=3,
                boundary_kernel=boundary_kernel,
                neighbor_kernel=neighbor_kernel,
                keep_ratio=keep_ratio,
                min_points=min_points,
            )  # 先拿到高质量边界点，再用于 bank 更新。

            update_boundary_prototype_bank_from_filtered_points(
                bank=bank,
                filtered_boundary_dict=filtered_boundary_dict,
                feature_already_normalized=False,
            )  # 直接调用刚写好的第二部分函数，做动态图 prototype bank 更新。

    if bank is None:  # 理论兜底分支。
        raise RuntimeError("Failed to build prototype bank from reference subset.")
    return bank


def visualize_boundary_prototype_positions(
    raw_boundary_dict: BoundaryDict,
    filtered_boundary_dict: BoundaryDict,
    feature_map: torch.Tensor,
    image_proto_dict,
    bank: DynamicBoundaryPrototypeBank,
    save_path: str | Path,
) -> None:
    """按边界类别可视化“当前类点簇 + 图级 prototype + EMA bank prototype”。

    每个边界类别画两张图：
    1. 左边：当前边界类别内部 PCA 投影，区分 raw / kept / dropped，并叠加图级和 bank prototype。
    2. 右边：当前边界类别内部 raw 点对图级/EMA bank prototype 的相似度分布。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_path).expanduser().resolve()  # 路径标准化。
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 输出目录不存在就创建。

    sorted_keys = sorted(raw_boundary_dict.keys())  # 只看当前图真正出现过的边界类别。
    if len(sorted_keys) == 0:  # 没边界点时直接返回。
        return

    num_rows = len(sorted_keys)  # 每个边界类别占一行。
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 5 * num_rows))  # 左边 PCA，右边 similarity 分布。
    if num_rows == 1:  # 单行时把 axes 统一成二维列表。
        axes = [axes]

    for row_index, boundary_key in enumerate(sorted_keys):  # 每个边界类别单独分析，这样问的问题才是对的。
        raw_data = raw_boundary_dict[boundary_key]  # 当前边界类 raw 数据。
        filtered_data = filtered_boundary_dict[boundary_key]  # 当前边界类 filtered 数据。

        raw_coords = raw_data["coords"]  # 当前类全部 raw 点坐标。
        kept_coords = filtered_data["coords"]  # 当前类保留点坐标。
        raw_features = gather_point_features(feature_map, raw_coords)  # 当前类 raw 点特征。
        normalized_raw_features = F.normalize(raw_features, p=2, dim=1, eps=1e-12)  # 用于 similarity 计算的归一化特征。

        image_entries = image_proto_dict.get(boundary_key, [])  # 当前图对应的图级 prototype 条目。
        if len(image_entries) == 0:  # 理论上不该发生，但还是兜一下。
            continue
        image_proto = image_entries[0]["image_proto"]  # 当前脚本 batch=1，所以这里只有一个图级 prototype。
        bank_proto = bank.get(boundary_key[0], boundary_key[1])  # 从动态 EMA bank 里取 prototype。
        if bank_proto is None:  # 当前类没注册进 bank 时就跳过。
            continue

        height = int(feature_map.size(2))  # 编码坐标要用 H。
        width = int(feature_map.size(3))  # 编码坐标要用 W。
        raw_code = encode_coords(raw_coords, height=height, width=width)  # raw 坐标编码。
        kept_code = encode_coords(kept_coords, height=height, width=width) if kept_coords.numel() > 0 else raw_code.new_zeros((0,))  # kept 坐标编码。
        keep_mask = torch.isin(raw_code, kept_code) if kept_code.numel() > 0 else torch.zeros_like(raw_code, dtype=torch.bool)  # 当前类哪些点被保留，先算出来。
        drop_mask = ~keep_mask  # 剩下的就是 dropped 点。

        kept_features = raw_features[keep_mask]  # kept 点特征。
        dropped_features = raw_features[drop_mask]  # dropped 点特征。
        image_similarity = torch.matmul(normalized_raw_features, image_proto.to(normalized_raw_features.device))  # raw 点对图级 prototype 的相似度。
        bank_similarity = torch.matmul(normalized_raw_features, bank_proto.to(normalized_raw_features.device))  # raw 点对 EMA bank prototype 的相似度。

        raw_proj, kept_proj, dropped_proj, image_proj, bank_proj = compute_class_pca_projection_with_bank(
            raw_features=raw_features,
            kept_features=kept_features,
            dropped_features=dropped_features,
            image_proto=image_proto,
            bank_proto=bank_proto,
        )  # 只在当前类内部做 PCA，并把图级/EMA 原型一起投进去。

        axis_pca = axes[row_index][0]  # 左图：类内特征空间。
        axis_hist = axes[row_index][1]  # 右图：相似度分布。

        if raw_proj.numel() > 0:
            axis_pca.scatter(
                raw_proj[:, 0].cpu().numpy(),
                raw_proj[:, 1].cpu().numpy(),
                s=12,
                c="lightgray",
                linewidths=0.0,
                alpha=0.45,
                label=f"raw={raw_proj.size(0)}",
            )  # 全部 raw 点先画成浅灰底层。
        if dropped_proj.numel() > 0:
            axis_pca.scatter(
                dropped_proj[:, 0].cpu().numpy(),
                dropped_proj[:, 1].cpu().numpy(),
                s=18,
                c="tab:red",
                linewidths=0.0,
                alpha=0.80,
                label=f"dropped={dropped_proj.size(0)}",
            )  # dropped 点单独用红色标出来，看它们是不是偏离主簇。
        if kept_proj.numel() > 0:
            axis_pca.scatter(
                kept_proj[:, 0].cpu().numpy(),
                kept_proj[:, 1].cpu().numpy(),
                s=18,
                c="tab:blue",
                linewidths=0.0,
                alpha=0.85,
                label=f"kept={kept_proj.size(0)}",
            )  # kept 点用蓝色，看筛选后主簇是不是更紧。
        axis_pca.scatter(
            image_proj[:, 0].cpu().numpy(),
            image_proj[:, 1].cpu().numpy(),
            s=140,
            c="gold",
            marker="*",
            edgecolors="black",
            linewidths=0.7,
            label="image_proto",
        )  # 当前图级 prototype 用金色星号画出来。
        if bank_proj is not None:
            axis_pca.scatter(
                bank_proj[:, 0].cpu().numpy(),
                bank_proj[:, 1].cpu().numpy(),
                s=120,
                c="limegreen",
                marker="P",
                edgecolors="black",
                linewidths=0.7,
                label="bank_proto",
            )  # EMA bank prototype 用绿色粗十字画出来，这就是你现在最关心的位置。
        axis_pca.set_title(f"{boundary_key_to_string(boundary_key)} PCA")  # 标题写清当前边界类。
        axis_pca.set_xlabel("PC-1")
        axis_pca.set_ylabel("PC-2")
        axis_pca.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        axis_pca.legend(loc="best", fontsize=8, framealpha=0.9)

        axis_hist.hist(
            image_similarity.cpu().numpy(),
            bins=24,
            color="lightgray",
            alpha=0.60,
            label="sim(raw, image_proto)",
        )  # raw 点对图级 prototype 的相似度分布。
        axis_hist.hist(
            bank_similarity.cpu().numpy(),
            bins=24,
            color="tab:green",
            alpha=0.45,
            label="sim(raw, bank_proto)",
        )  # raw 点对 EMA bank prototype 的相似度分布。
        if keep_mask.any():
            axis_hist.hist(
                image_similarity[keep_mask].cpu().numpy(),
                bins=24,
                color="tab:blue",
                alpha=0.55,
                label=f"kept={int(keep_mask.sum())}",
            )  # kept 点相似度再叠一层，便于判断它们是否集中在高相似度区。
        if drop_mask.any():
            axis_hist.hist(
                image_similarity[drop_mask].cpu().numpy(),
                bins=24,
                color="tab:red",
                alpha=0.55,
                label=f"dropped={int(drop_mask.sum())}",
            )  # dropped 点相似度再叠一层，便于判断它们是否更多落在低相似度尾部。
        axis_hist.set_title(f"{boundary_key_to_string(boundary_key)} Similarity")
        axis_hist.set_xlabel("Cosine Similarity")
        axis_hist.set_ylabel("Count")
        axis_hist.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        axis_hist.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.tight_layout()  # 自动整理排版。
    fig.savefig(save_path, dpi=220, bbox_inches="tight")  # 保存“原型位置检查图”。
    plt.close(fig)  # 及时关图。


def build_prompt_visualization_case(
    device: torch.device,
    se_source: Encoder,
    seg_decoder: Decoder,
    bank: DynamicBoundaryPrototypeBank,
    patient_index: int,
    slice_index: Optional[int],
    boundary_kernel: int,
    neighbor_kernel: int,
    point_score_threshold: float = 0.65,
    point_topk_per_boundary: int = 1,
    box_quantile: float = 0.80,
) -> Dict[str, object]:
    """为单张 PICAI 切片构建第三步 prompt 可视化所需数据。

    当前实现采用下面这套规则：
        1. 用分割 logits 的 argmax 作为伪标签。
        2. 用伪标签提取 ordered 边界候选区域。
        3. 只在伪标签的 ordered 候选区域内，计算 ordered boundary prototype score。
        4. point prompt:
           - 对每个 ordered boundary key 保留最高分 seed。
        5. box prompt:
           - 对同一条边界对的两侧高分区域取并集，再生成一个 pair-level box。
    """
    image_batch, gt_seg_batch, image_2d, used_patient_index, used_slice_index = load_picai_source_slice(
        patient_index=patient_index,
        slice_index=slice_index,
    )  # 当前切片的图像与 GT 标签先读出来。

    seg_logits, feature_map = extract_pretrained_logits_and_feature_map(
        image_batch=image_batch,
        se_source=se_source,
        seg_decoder=seg_decoder,
        device=device,
    )  # 同时提 pseudo logits 和边界特征图。
    pseudo_label = torch.argmax(seg_logits, dim=1).to(torch.long)  # 伪标签直接用 argmax，和 teacher pseudo label 的基本形式一致。

    pseudo_boundary_dict = assign_fine_boundary_labels(
        seg_label=pseudo_label,
        num_classes=int(seg_logits.size(1)),
        boundary_kernel=boundary_kernel,
        neighbor_kernel=neighbor_kernel,
    )  # 只用伪标签做第三步，这才符合“目标域 prompt 由伪标签驱动”的设计。
    candidate_mask_dict = build_boundary_mask_dict_from_boundary_dict(
        boundary_dict=pseudo_boundary_dict,
        batch_size=int(pseudo_label.size(0)),
        height=int(pseudo_label.size(1)),
        width=int(pseudo_label.size(2)),
    )  # ordered boundary 候选坐标转成 mask，后面限制 score 计算区域。

    score_dict = compute_all_ordered_boundary_scores(
        feature_map=feature_map,
        prototype_source=bank,
        target_keys=sorted(pseudo_boundary_dict.keys()),
        candidate_mask_dict=candidate_mask_dict,
        fill_value=-1.0,
    )  # 每个 ordered key 都单独算一张 score 图，(A,B) 与 (B,A) 绝不合并。
    point_prompt_dict = extract_ordered_boundary_prompt_seeds(
        score_dict=score_dict,
        topk_per_boundary=point_topk_per_boundary,
        min_score=point_score_threshold,
    )  # point prompt 保持 ordered，每个方向独立取最高分点。
    box_prompt_dict = build_pair_box_prompt_dict_from_score_dict(
        score_dict=score_dict,
        candidate_mask_dict=candidate_mask_dict,
        box_quantile=box_quantile,
        min_pixels=4,
        padding=6,
    )  # box prompt 用 pair-level 区域，便于看整条冲突边界的局部范围。

    return {
        "patient_index": int(used_patient_index),
        "slice_index": int(used_slice_index),
        "image_2d": image_2d,
        "pseudo_label": pseudo_label[0].clone(),
        "gt_label": gt_seg_batch[0].clone(),
        "point_prompt_dict": point_prompt_dict,
        "box_prompt_dict": box_prompt_dict,
        "score_dict": score_dict,
    }


def visualize_prompt_grid(
    case_records: List[Dict[str, object]],
    label_key: str,
    save_path: str | Path,
) -> None:
    """把多张切片的 point / box prompt 画成一个网格总览图。

    输入：
        case_records: list
            每个元素都来自 `build_prompt_visualization_case`。
        label_key: str
            背景标签类型：
            - "pseudo_label"
            - "gt_label"
        save_path: str | Path
            输出图像路径。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if label_key not in {"pseudo_label", "gt_label"}:
        raise ValueError("label_key must be either 'pseudo_label' or 'gt_label'")

    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(case_records) == 0:
        return

    ordered_keys = sorted({
        tuple(seed["ordered_boundary_key"])
        for case_record in case_records
        for seed_list in case_record["point_prompt_dict"].values()
        for seed in seed_list
    })  # 所有图里真正出现过的 ordered point key 先汇总出来。
    pair_keys = sorted({
        pair_key
        for case_record in case_records
        for pair_key in case_record["box_prompt_dict"].keys()
    })  # box prompt 的 pair key 也汇总出来。

    point_colors = get_boundary_colors(list(ordered_keys)) if len(ordered_keys) > 0 else {}
    pair_colors = get_boundary_colors([tuple(pair_key) for pair_key in pair_keys]) if len(pair_keys) > 0 else {}

    num_cases = len(case_records)
    num_cols = 2 if num_cases > 1 else 1
    num_rows = math.ceil(num_cases / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7.0 * num_cols, 6.5 * num_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, case_record in zip(axes, case_records):
        label_2d = case_record[label_key]
        label_np = label_2d.detach().cpu().numpy()
        axis.imshow(label_np, cmap="nipy_spectral", interpolation="nearest")
        axis.set_title(
            f"patient={int(case_record['patient_index']):03d}, "
            f"slice={int(case_record['slice_index']):02d}, "
            f"background={'pseudo' if label_key == 'pseudo_label' else 'gt'}"
        )
        axis.axis("off")

        for pair_key, box_info in sorted(case_record["box_prompt_dict"].items()):  # pair-level box 先画出来。
            x1, y1, x2, y2 = box_info["box"]
            rect = Rectangle(
                (x1, y1),
                x2 - x1 + 1,
                y2 - y1 + 1,
                fill=False,
                edgecolor=pair_colors[tuple(pair_key)],
                linewidth=2.0,
                linestyle="-",
                alpha=0.95,
            )
            axis.add_patch(rect)
            axis.text(
                x1,
                max(0, y1 - 2),
                f"box{tuple(pair_key)}",
                color=pair_colors[tuple(pair_key)],
                fontsize=8,
                ha="left",
                va="bottom",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 1},
            )  # box 左上角顺手写一下 pair key，便于人工核对。

        for boundary_key, seed_list in sorted(case_record["point_prompt_dict"].items()):  # ordered point prompt 再画上去。
            for seed in seed_list:
                x = int(seed["x"])
                y = int(seed["y"])
                axis.scatter(
                    x,
                    y,
                    s=80,
                    c=[point_colors[boundary_key]],
                    marker="o",
                    edgecolors="white",
                    linewidths=0.9,
                    alpha=0.95,
                )
                axis.text(
                    x + 1,
                    y + 1,
                    boundary_key_to_string(boundary_key),
                    color=point_colors[boundary_key],
                    fontsize=8,
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "black", "alpha": 0.30, "pad": 1},
                )  # 每个点 prompt 都直接标出 ordered key，这样一眼就能看左右方向有没有搞反。

    for axis in axes[num_cases:]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_picai_pretrained_prompt_visualization(
    save_dir: str | Path,
    eval_patient_indices: Tuple[int, ...] = (0, 1, 2, 3),
    eval_slice_rank: int = 0,
    boundary_kernel: int = 5,
    neighbor_kernel: int = 3,
    keep_ratio: float = 0.8,
    min_points: int = 10,
    bank_patient_indices: Tuple[int, ...] = (0, 1, 2, 3),
    bank_slice_ranks: Tuple[int, ...] = (0, 1),
    point_score_threshold: float = 0.65,
    point_topk_per_boundary: int = 1,
    box_quantile: float = 0.80,
) -> List[Dict[str, object]]:
    """在多张 PICAI 切片上可视化第三步 prompt 生成结果。"""
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    se_source, seg_decoder = load_reference_source_models(device=device)  # 继续沿用和参考训练脚本一致的权重读取方式。

    bank = build_boundary_prototype_bank_from_reference_subset(
        device=device,
        se_source=se_source,
        seg_decoder=seg_decoder,
        patient_indices=bank_patient_indices,
        slice_ranks=bank_slice_ranks,
        boundary_kernel=boundary_kernel,
        neighbor_kernel=neighbor_kernel,
        keep_ratio=keep_ratio,
        min_points=min_points,
    )  # 先用少量源域真实切片构建 ordered boundary prototype bank。

    case_records: List[Dict[str, object]] = []
    for patient_index in eval_patient_indices:  # 多挑几张 PICAI 切片，集中看 prompt 是否合理。
        dataset = PICAITrainDataset(
            data_root=str(Path("/home/chenxu/dataset/picai").expanduser().resolve() / "train"),
            modality="t2w",
            n_slices=1,
        )
        slice_index = choose_slice_index_from_seg_volume(
            dataset.data_file["seg"][int(patient_index)],
            top_rank=eval_slice_rank,
        )
        case_record = build_prompt_visualization_case(
            device=device,
            se_source=se_source,
            seg_decoder=seg_decoder,
            bank=bank,
            patient_index=int(patient_index),
            slice_index=int(slice_index),
            boundary_kernel=boundary_kernel,
            neighbor_kernel=neighbor_kernel,
            point_score_threshold=point_score_threshold,
            point_topk_per_boundary=point_topk_per_boundary,
            box_quantile=box_quantile,
        )
        case_records.append(case_record)

    pseudo_prompt_path = save_dir / "picai_prompt_on_pseudo_labels.png"
    gt_prompt_path = save_dir / "picai_prompt_on_gt_labels.png"
    visualize_prompt_grid(case_records=case_records, label_key="pseudo_label", save_path=pseudo_prompt_path)
    visualize_prompt_grid(case_records=case_records, label_key="gt_label", save_path=gt_prompt_path)

    print("=" * 80)
    print("PICAI Prompt Visualization")
    print("=" * 80)
    print(f"device={device}")
    print(f"eval_patient_indices={eval_patient_indices}, eval_slice_rank={eval_slice_rank}")
    print(f"boundary_kernel={boundary_kernel}, neighbor_kernel={neighbor_kernel}")
    print(f"point_score_threshold={point_score_threshold}, point_topk_per_boundary={point_topk_per_boundary}, box_quantile={box_quantile}")
    for case_record in case_records:
        point_count = sum(len(seed_list) for seed_list in case_record["point_prompt_dict"].values())
        box_count = len(case_record["box_prompt_dict"])
        print(
            f"patient={int(case_record['patient_index']):03d}, "
            f"slice={int(case_record['slice_index']):02d} | "
            f"point_prompts={point_count:2d} | "
            f"box_prompts={box_count:2d}"
        )
    print("-" * 80)
    print(f"pseudo-label prompt visualization saved to: {pseudo_prompt_path}")
    print(f"gt-label prompt visualization saved to:     {gt_prompt_path}")
    return case_records


def run_picai_pretrained_boundary_visualization(
    save_dir: str | Path,
    patient_index: int = 0,
    slice_index: Optional[int] = None,
    boundary_kernel: int = 5,
    neighbor_kernel: int = 3,
    keep_ratio: float = 0.8,
    min_points: int = 10,
    bank_patient_indices: Tuple[int, ...] = (0, 1, 2, 3),
    bank_slice_ranks: Tuple[int, ...] = (0, 1),
) -> Dict[BoundaryKey, Dict[str, int | float | bool]]:
    """在真实 PICAI 切片上使用预训练网络特征和动态 bank 做可视化。"""
    save_dir = Path(save_dir).expanduser().resolve()  # 输出目录标准化。
    save_dir.mkdir(parents=True, exist_ok=True)  # 输出目录不存在就建好。

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有 CUDA 就用，没有也照样能跑。
    se_source, seg_decoder = load_reference_source_models(device=device)  # 按参考训练脚本加载源域模型和权重。

    bank = build_boundary_prototype_bank_from_reference_subset(
        device=device,
        se_source=se_source,
        seg_decoder=seg_decoder,
        patient_indices=bank_patient_indices,
        slice_ranks=bank_slice_ranks,
        boundary_kernel=boundary_kernel,
        neighbor_kernel=neighbor_kernel,
        keep_ratio=keep_ratio,
        min_points=min_points,
    )  # 先从少量真实 PICAI 切片构建一个动态 EMA bank。

    image_batch, seg_batch, image_2d, used_patient_index, used_slice_index = load_picai_source_slice(
        patient_index=patient_index,
        slice_index=slice_index,
    )  # 读取当前要可视化的真实源域切片。
    feature_map = extract_pretrained_feature_map(
        image_batch=image_batch,
        se_source=se_source,
        seg_decoder=seg_decoder,
        device=device,
    )  # 用真实预训练网络提特征。

    raw_boundary_dict, filtered_boundary_dict, summary_info = build_source_fine_boundary_points(
        feature_map=feature_map,
        seg_label=seg_batch,
        num_classes=3,
        boundary_kernel=boundary_kernel,
        neighbor_kernel=neighbor_kernel,
        keep_ratio=keep_ratio,
        min_points=min_points,
    )  # 调用第一部分工具函数，得到 raw / filtered 边界点结果。
    image_proto_dict = compute_image_level_boundary_prototypes(
        filtered_boundary_dict=filtered_boundary_dict,
        feature_already_normalized=False,
    )  # 调用第二部分工具函数，得到当前图像的图级 prototype。

    raw_path = save_dir / f"picai_pretrained_raw_patient{used_patient_index:03d}_slice{used_slice_index:02d}.png"  # raw 点图路径。
    filtered_path = save_dir / f"picai_pretrained_filtered_patient{used_patient_index:03d}_slice{used_slice_index:02d}.png"  # filtered 点图路径。
    dashboard_path = save_dir / f"picai_pretrained_dashboard_patient{used_patient_index:03d}_slice{used_slice_index:02d}.png"  # 几何空间综合图路径。
    prototype_path = save_dir / f"picai_pretrained_prototype_position_patient{used_patient_index:03d}_slice{used_slice_index:02d}.png"  # 原型位置图路径。

    visualize_fine_boundary_points(seg_label_2d=seg_batch[0], boundary_dict=raw_boundary_dict, save_path=raw_path)  # 原始边界点几何图。
    visualize_fine_boundary_points(seg_label_2d=seg_batch[0], boundary_dict=filtered_boundary_dict, save_path=filtered_path)  # 净化后边界点几何图。
    visualize_fine_boundary_dashboard(
        image_2d=image_2d,
        seg_label_2d=seg_batch[0],
        raw_boundary_dict=raw_boundary_dict,
        filtered_boundary_dict=filtered_boundary_dict,
        save_path=dashboard_path,
    )  # 原图、标签、raw/filter 几何空间综合图。
    visualize_boundary_prototype_positions(
        raw_boundary_dict=raw_boundary_dict,
        filtered_boundary_dict=filtered_boundary_dict,
        feature_map=feature_map,
        image_proto_dict=image_proto_dict,
        bank=bank,
        save_path=prototype_path,
    )  # 关键图：当前类点簇 + 图级 prototype + EMA bank prototype。

    print("=" * 80)
    print("PICAI Pretrained Boundary + Prototype Visualization")
    print("=" * 80)
    print(f"device={device}, patient_index={used_patient_index}, slice_index={used_slice_index}")
    print(f"boundary_kernel={boundary_kernel}, neighbor_kernel={neighbor_kernel}, keep_ratio={keep_ratio}, min_points={min_points}")
    print(f"bank_patient_indices={bank_patient_indices}, bank_slice_ranks={bank_slice_ranks}")
    for boundary_key in sorted(summary_info.keys()):
        stats = summary_info[boundary_key]
        print(
            f"boundary={boundary_key_to_string(boundary_key):>8s} | "
            f"raw={int(stats['raw_count']):4d} | "
            f"kept={int(stats['kept_count']):4d} | "
            f"actual_keep_ratio={float(stats['actual_keep_ratio']):.3f} | "
            f"skip_filter={bool(stats['skip_filter'])}"
        )
    print("-" * 80)
    print(f"raw visualization saved to:        {raw_path}")
    print(f"filtered visualization saved to:   {filtered_path}")
    print(f"dashboard visualization saved to:  {dashboard_path}")
    print(f"prototype visualization saved to:  {prototype_path}")
    return summary_info


def _prototype_exists(
    prototype_library: DynamicBoundaryPrototypeBank | Dict[BoundaryKey, object],
    a: int,
    b: int,
) -> bool:
    """检查某个 ordered key 的 prototype 是否存在。"""
    boundary_key = canonicalize_ordered_boundary_key(a, b)
    if isinstance(prototype_library, DynamicBoundaryPrototypeBank):
        return prototype_library.get(boundary_key[0], boundary_key[1]) is not None
    if not isinstance(prototype_library, dict):
        raise TypeError("prototype_library must be a DynamicBoundaryPrototypeBank or a dictionary")
    if boundary_key not in prototype_library:
        return False
    entry = prototype_library[boundary_key]
    if isinstance(entry, torch.Tensor):
        return True
    if isinstance(entry, dict):
        return isinstance(entry.get("prototype"), torch.Tensor)
    return False


def build_pair_boxes_from_boundary_dict(
    boundary_dict: BoundaryDict,
    height: int,
    width: int,
    padding: int = 6,
    require_bidirectional: bool = True,
    prototype_library: Optional[DynamicBoundaryPrototypeBank | Dict[BoundaryKey, object]] = None,
) -> List[Dict[str, object]]:
    """从 ordered boundary 坐标直接构造 pair-level 局部 box。

    输入：
        boundary_dict: dict
            `assign_fine_boundary_labels` 的输出，至少包含：
            `{(A, B): {"coords": Tensor[N, 3], ...}}`
        height: int
            图像高度。
        width: int
            图像宽度。
        padding: int
            box 四周额外扩展的像素。
        require_bidirectional: bool
            是否要求同一对类别同时出现 `(A, B)` 与 `(B, A)`。
        prototype_library: Optional[...]
            若给出，则进一步要求 box 对应的两个方向 prototype 都已存在。

    输出：
        box_list: list
            每个元素形式为：
            {
                "batch_idx": int,
                "a": int,
                "b": int,
                "box": (x1, y1, x2, y2),
                "ordered_keys": ((A, B), (B, A)),
                "num_pixels": int,
            }

    说明：
        1. 当前 helper 直接从 label 的 ordered 边界点坐标回推局部 box。
        2. 这里不再依赖旧版第三部分的全图 score map。
        3. 默认只保留真正具备双向 ordered 定义的 pair，这样后续才能稳定找出两个核心点。
    """
    pair_coord_buckets: Dict[Tuple[int, int, int], List[torch.Tensor]] = {}
    pair_ordered_key_buckets: Dict[Tuple[int, int, int], set[BoundaryKey]] = {}

    for boundary_key, boundary_data in boundary_dict.items():
        coords = boundary_data.get("coords")
        if not isinstance(coords, torch.Tensor):
            raise TypeError(f"boundary_dict[{boundary_key}]['coords'] must be a torch.Tensor")
        if coords.numel() == 0:
            continue

        coords = coords.long()
        pair_key = tuple(sorted((int(boundary_key[0]), int(boundary_key[1]))))
        unique_batch_indices = torch.unique(coords[:, 0])
        for batch_idx in unique_batch_indices.tolist():
            batch_coords = coords[coords[:, 0] == int(batch_idx)]
            if batch_coords.numel() == 0:
                continue
            bucket_key = (int(batch_idx), int(pair_key[0]), int(pair_key[1]))
            pair_coord_buckets.setdefault(bucket_key, []).append(batch_coords)
            pair_ordered_key_buckets.setdefault(bucket_key, set()).add(
                canonicalize_ordered_boundary_key(boundary_key[0], boundary_key[1])
            )

    box_list: List[Dict[str, object]] = []
    for bucket_key in sorted(pair_coord_buckets.keys()):
        batch_idx, a, b = bucket_key
        expected_ordered_keys = {
            canonicalize_ordered_boundary_key(a, b),
            canonicalize_ordered_boundary_key(b, a),
        }
        present_ordered_keys = pair_ordered_key_buckets[bucket_key]
        if require_bidirectional and not expected_ordered_keys.issubset(present_ordered_keys):
            continue

        if prototype_library is not None:
            has_ab = _prototype_exists(prototype_library, a, b)
            has_ba = _prototype_exists(prototype_library, b, a)
            if require_bidirectional and not (has_ab and has_ba):
                continue
            if not require_bidirectional and not (has_ab or has_ba):
                continue

        merged_coords = torch.cat(pair_coord_buckets[bucket_key], dim=0)
        ys = merged_coords[:, 1]
        xs = merged_coords[:, 2]
        x1 = max(0, int(xs.min().item()) - int(padding))
        y1 = max(0, int(ys.min().item()) - int(padding))
        x2 = min(int(width) - 1, int(xs.max().item()) + int(padding))
        y2 = min(int(height) - 1, int(ys.max().item()) + int(padding))

        box_list.append(
            {
                "batch_idx": int(batch_idx),
                "a": int(a),
                "b": int(b),
                "box": (x1, y1, x2, y2),
                "ordered_keys": tuple(sorted(expected_ordered_keys)),
                "num_pixels": int(merged_coords.size(0)),
            }
        )

    return box_list


def build_ordered_core_visualization_case(
    device: torch.device,
    se_source: Encoder,
    seg_decoder: Decoder,
    bank: DynamicBoundaryPrototypeBank,
    patient_index: int,
    slice_index: Optional[int],
    boundary_kernel: int,
    neighbor_kernel: int,
    box_padding: int = 6,
    use_soft_uncertainty: bool = False,
) -> Dict[str, object]:
    """为单张 PICAI 切片构建“第三部分：box 内双向核心点”可视化数据。

    当前实现明确分成两条平行路径：
        1. pseudo-label 几何路径：
           pseudo_label -> ordered boundary -> pair box -> core_ab / core_ba
        2. gt-label 几何路径：
           gt_label -> ordered boundary -> pair box -> core_ab / core_ba

    这样你可以直接比较：
        - 在伪标签几何约束下，box 和双向核心点落在哪里
        - 在真实标签几何约束下，box 和双向核心点落在哪里
    """
    image_batch, gt_seg_batch, image_2d, used_patient_index, used_slice_index = load_picai_source_slice(
        patient_index=patient_index,
        slice_index=slice_index,
    )

    seg_logits, feature_map = extract_pretrained_logits_and_feature_map(
        image_batch=image_batch,
        se_source=se_source,
        seg_decoder=seg_decoder,
        device=device,
    )
    pseudo_label = torch.argmax(seg_logits, dim=1).to(torch.long)
    prob_map = torch.softmax(seg_logits.to(torch.float32), dim=1)

    pseudo_boundary_dict = assign_fine_boundary_labels(
        seg_label=pseudo_label,
        num_classes=int(seg_logits.size(1)),
        boundary_kernel=boundary_kernel,
        neighbor_kernel=neighbor_kernel,
    )
    gt_boundary_dict = assign_fine_boundary_labels(
        seg_label=gt_seg_batch,
        num_classes=int(seg_logits.size(1)),
        boundary_kernel=boundary_kernel,
        neighbor_kernel=neighbor_kernel,
    )

    height = int(pseudo_label.size(1))
    width = int(pseudo_label.size(2))
    pseudo_box_list = build_pair_boxes_from_boundary_dict(
        boundary_dict=pseudo_boundary_dict,
        height=height,
        width=width,
        padding=box_padding,
        require_bidirectional=True,
        prototype_library=bank,
    )
    gt_box_list = build_pair_boxes_from_boundary_dict(
        boundary_dict=gt_boundary_dict,
        height=height,
        width=width,
        padding=box_padding,
        require_bidirectional=True,
        prototype_library=bank,
    )

    pseudo_core_results = generate_ordered_core_points_for_boxes(
        feature_map=feature_map,
        box_list=pseudo_box_list,
        prototype_library=bank,
        prob_map=prob_map,
        use_soft_uncertainty=use_soft_uncertainty,
    )
    gt_core_results = generate_ordered_core_points_for_boxes(
        feature_map=feature_map,
        box_list=gt_box_list,
        prototype_library=bank,
        prob_map=prob_map,
        use_soft_uncertainty=use_soft_uncertainty,
    )

    return {
        "patient_index": int(used_patient_index),
        "slice_index": int(used_slice_index),
        "image_2d": image_2d,
        "pseudo_label": pseudo_label[0].clone(),
        "gt_label": gt_seg_batch[0].clone(),
        "pseudo_box_list": pseudo_box_list,
        "gt_box_list": gt_box_list,
        "pseudo_core_results": pseudo_core_results,
        "gt_core_results": gt_core_results,
    }


def build_box_core_list_from_core_results(
    core_result_list: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """把第三部分的 core result 列表整理成第四部分可直接消费的输入列表。

    输入：
        core_result_list: list
            `generate_ordered_core_points_for_boxes` 的输出列表。

    输出：
        box_core_list: list
            每个元素至少包含：
            - `box`
            - `core_ab`
            - `core_ba`
            - `a`
            - `b`

    说明：
        1. 当前第四部分必须同时拿到 `core_ab` 和 `core_ba`，否则就没有稳定的跨边界方向。
        2. 所以这里只保留双向核心点都存在的 pair。
    """
    box_core_list: List[Dict[str, object]] = []
    for result in core_result_list:
        core_ab = result.get("core_ab")
        core_ba = result.get("core_ba")
        pair = result.get("pair")
        box = result.get("box")
        if not isinstance(core_ab, dict) or not isinstance(core_ba, dict):
            continue
        if not isinstance(pair, tuple) or len(pair) != 2:
            continue
        if not isinstance(box, dict):
            continue
        box_core_list.append(
            {
                "box": box,
                "core_ab": core_ab,
                "core_ba": core_ba,
                "a": int(pair[0]),
                "b": int(pair[1]),
            }
        )
    return box_core_list


def build_ordered_point_prompt_visualization_case(
    device: torch.device,
    se_source: Encoder,
    seg_decoder: Decoder,
    bank: DynamicBoundaryPrototypeBank,
    patient_index: int,
    slice_index: Optional[int],
    boundary_kernel: int,
    neighbor_kernel: int,
    box_padding: int = 6,
    use_soft_uncertainty: bool = False,
    offset_distance: float = 3.0,
    window_radius: int = 12,
    sigma: float = 5.0,
    topk_per_side: int = 3,
    min_distance: float = 5.0,
    min_score: Optional[float] = None,
) -> Dict[str, object]:
    """为单张 PICAI 切片构建“第三部分 + 第四部分”联合可视化数据。

    当前实现流程：
        1. 先走第三部分：
           pseudo/gt label -> pair box -> core_ab/core_ba
        2. 再接第四部分：
           core_ab/core_ba -> direction -> ref centers -> points_a/points_b

    这样你可以直接看：
        - 伪标签几何下的双向核心点和多点 prompt 分布
        - 真实标签几何下的双向核心点和多点 prompt 分布
    """
    case_record = build_ordered_core_visualization_case(
        device=device,
        se_source=se_source,
        seg_decoder=seg_decoder,
        bank=bank,
        patient_index=patient_index,
        slice_index=slice_index,
        boundary_kernel=boundary_kernel,
        neighbor_kernel=neighbor_kernel,
        box_padding=box_padding,
        use_soft_uncertainty=use_soft_uncertainty,
    )

    image_batch, _, _, _, _ = load_picai_source_slice(
        patient_index=int(case_record["patient_index"]),
        slice_index=int(case_record["slice_index"]),
    )
    seg_logits, _ = extract_pretrained_logits_and_feature_map(
        image_batch=image_batch,
        se_source=se_source,
        seg_decoder=seg_decoder,
        device=device,
    )
    prob_map = torch.softmax(seg_logits.to(torch.float32), dim=1)

    pseudo_box_core_list = build_box_core_list_from_core_results(case_record["pseudo_core_results"])
    gt_box_core_list = build_box_core_list_from_core_results(case_record["gt_core_results"])

    pseudo_point_prompt_results = generate_point_prompts_for_box_list(
        prob_map=prob_map,
        box_core_list=pseudo_box_core_list,
        offset_distance=offset_distance,
        window_radius=window_radius,
        sigma=sigma,
        topk_per_side=topk_per_side,
        min_distance=min_distance,
        min_score=min_score,
    )
    gt_point_prompt_results = generate_point_prompts_for_box_list(
        prob_map=prob_map,
        box_core_list=gt_box_core_list,
        offset_distance=offset_distance,
        window_radius=window_radius,
        sigma=sigma,
        topk_per_side=topk_per_side,
        min_distance=min_distance,
        min_score=min_score,
    )

    case_record["pseudo_point_prompt_results"] = pseudo_point_prompt_results
    case_record["gt_point_prompt_results"] = gt_point_prompt_results
    return case_record


def visualize_ordered_core_grid(
    case_records: List[Dict[str, object]],
    label_key: str,
    result_key: str,
    save_path: str | Path,
) -> None:
    """把多张切片的 ordered core point 几何分布画成网格图。

    输入：
        case_records: list
            每个元素都来自 `build_ordered_core_visualization_case`。
        label_key: str
            背景标签类型：
            - `pseudo_label`
            - `gt_label`
        result_key: str
            当前要显示的核心点结果：
            - `pseudo_core_results`
            - `gt_core_results`
        save_path: str | Path
            输出图像路径。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if label_key not in {"pseudo_label", "gt_label"}:
        raise ValueError("label_key must be either 'pseudo_label' or 'gt_label'")
    if result_key not in {"pseudo_core_results", "gt_core_results"}:
        raise ValueError("result_key must be either 'pseudo_core_results' or 'gt_core_results'")

    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(case_records) == 0:
        return

    pair_keys = sorted({
        tuple(result["pair"])
        for case_record in case_records
        for result in case_record[result_key]
    })
    pair_colors = get_boundary_colors([tuple(pair_key) for pair_key in pair_keys]) if len(pair_keys) > 0 else {}

    num_cases = len(case_records)
    num_cols = 2 if num_cases > 1 else 1
    num_rows = math.ceil(num_cases / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7.2 * num_cols, 6.8 * num_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, case_record in zip(axes, case_records):
        label_2d = case_record[label_key]
        label_np = label_2d.detach().cpu().numpy()
        axis.imshow(label_np, cmap="nipy_spectral", interpolation="nearest")
        axis.set_title(
            f"patient={int(case_record['patient_index']):03d}, "
            f"slice={int(case_record['slice_index']):02d}, "
            f"geometry={'pseudo' if result_key == 'pseudo_core_results' else 'gt'}"
        )
        axis.axis("off")

        for result in case_record[result_key]:
            pair = tuple(result["pair"])
            box_info = result["box"]
            x1, y1, x2, y2 = box_info["x1"], box_info["y1"], box_info["x2"], box_info["y2"]
            box_color = pair_colors.get(pair, (1.0, 1.0, 0.0, 1.0))

            rect = Rectangle(
                (x1, y1),
                x2 - x1 + 1,
                y2 - y1 + 1,
                fill=False,
                edgecolor=box_color,
                linewidth=2.0,
                linestyle="-",
                alpha=0.95,
            )
            axis.add_patch(rect)
            axis.text(
                x1,
                max(0, y1 - 2),
                f"pair{pair}",
                color=box_color,
                fontsize=8,
                ha="left",
                va="bottom",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 1},
            )

            core_ab = result["core_ab"]
            core_ba = result["core_ba"]
            if isinstance(core_ab, dict):
                axis.scatter(
                    int(core_ab["x"]),
                    int(core_ab["y"]),
                    s=80,
                    c="red",
                    marker="o",
                    edgecolors="white",
                    linewidths=0.9,
                    alpha=0.95,
                )
                axis.text(
                    int(core_ab["x"]) + 1,
                    int(core_ab["y"]) + 1,
                    f"{pair[0]}->{pair[1]}",
                    color="red",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "white", "alpha": 0.55, "pad": 1},
                )
            if isinstance(core_ba, dict):
                axis.scatter(
                    int(core_ba["x"]),
                    int(core_ba["y"]),
                    s=95,
                    c="cyan",
                    marker="x",
                    linewidths=2.0,
                    alpha=0.95,
                )
                axis.text(
                    int(core_ba["x"]) + 1,
                    int(core_ba["y"]) - 1,
                    f"{pair[1]}->{pair[0]}",
                    color="cyan",
                    fontsize=8,
                    ha="left",
                    va="top",
                    bbox={"facecolor": "black", "alpha": 0.35, "pad": 1},
                )

    for axis in axes[num_cases:]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def visualize_point_prompt_grid(
    case_records: List[Dict[str, object]],
    label_key: str,
    result_key: str,
    save_path: str | Path,
) -> None:
    """把多张切片的第四部分 point prompts 画成一个网格总览图。

    输入：
        case_records: list
            每个元素都来自 `build_ordered_point_prompt_visualization_case`。
        label_key: str
            背景标签类型：
            - `pseudo_label`
            - `gt_label`
        result_key: str
            当前要显示的第四部分结果：
            - `pseudo_point_prompt_results`
            - `gt_point_prompt_results`
        save_path: str | Path
            输出图像路径。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if label_key not in {"pseudo_label", "gt_label"}:
        raise ValueError("label_key must be either 'pseudo_label' or 'gt_label'")
    if result_key not in {"pseudo_point_prompt_results", "gt_point_prompt_results"}:
        raise ValueError("result_key must be either 'pseudo_point_prompt_results' or 'gt_point_prompt_results'")

    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(case_records) == 0:
        return

    pair_keys = sorted({
        (int(result["a"]), int(result["b"]))
        for case_record in case_records
        for result in case_record[result_key]
    })
    pair_colors = get_boundary_colors([tuple(pair_key) for pair_key in pair_keys]) if len(pair_keys) > 0 else {}

    num_cases = len(case_records)
    num_cols = 2 if num_cases > 1 else 1
    num_rows = math.ceil(num_cases / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7.6 * num_cols, 7.0 * num_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, case_record in zip(axes, case_records):
        label_2d = case_record[label_key]
        label_np = label_2d.detach().cpu().numpy()
        axis.imshow(label_np, cmap="nipy_spectral", interpolation="nearest")
        axis.set_title(
            f"patient={int(case_record['patient_index']):03d}, "
            f"slice={int(case_record['slice_index']):02d}, "
            f"geometry={'pseudo' if result_key == 'pseudo_point_prompt_results' else 'gt'}"
        )
        axis.axis("off")

        for result in case_record[result_key]:
            a = int(result["a"])
            b = int(result["b"])
            pair = (a, b)
            box_info = result["box"]
            x_min = int(box_info["x_min"])
            y_min = int(box_info["y_min"])
            x_max = int(box_info["x_max"])
            y_max = int(box_info["y_max"])
            pair_color = pair_colors.get(pair, (1.0, 1.0, 0.0, 1.0))

            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min + 1,
                y_max - y_min + 1,
                fill=False,
                edgecolor=pair_color,
                linewidth=2.0,
                linestyle="-",
                alpha=0.95,
            )
            axis.add_patch(rect)
            axis.text(
                x_min,
                max(0, y_min - 2),
                f"pair{pair}",
                color=pair_color,
                fontsize=8,
                ha="left",
                va="bottom",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 1},
            )

            core_ab = result["core_ab"]
            core_ba = result["core_ba"]
            if isinstance(core_ab, dict):
                axis.scatter(
                    int(core_ab["x"]),
                    int(core_ab["y"]),
                    s=110,
                    c="red",
                    marker="*",
                    edgecolors="white",
                    linewidths=0.8,
                    alpha=0.95,
                )
            if isinstance(core_ba, dict):
                axis.scatter(
                    int(core_ba["x"]),
                    int(core_ba["y"]),
                    s=110,
                    c="cyan",
                    marker="*",
                    edgecolors="black",
                    linewidths=0.8,
                    alpha=0.95,
                )

            ref_center_a = result["ref_center_a"]
            ref_center_b = result["ref_center_b"]
            axis.scatter(
                [float(ref_center_a["x"])],
                [float(ref_center_a["y"])],
                s=85,
                facecolors="none",
                edgecolors="orange",
                linewidths=2.0,
                marker="o",
            )
            axis.scatter(
                [float(ref_center_b["x"])],
                [float(ref_center_b["y"])],
                s=85,
                c="lime",
                linewidths=2.0,
                marker="x",
            )

            for point in result["points_a"]:
                axis.scatter(
                    int(point["x"]),
                    int(point["y"]),
                    s=72,
                    c="orange",
                    marker="o",
                    edgecolors="black",
                    linewidths=0.6,
                    alpha=0.95,
                )
            for point in result["points_b"]:
                axis.scatter(
                    int(point["x"]),
                    int(point["y"]),
                    s=72,
                    c="lime",
                    marker="x",
                    linewidths=2.0,
                    alpha=0.95,
                )

    for axis in axes[num_cases:]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_picai_pretrained_ordered_core_visualization(
    save_dir: str | Path,
    eval_patient_indices: Tuple[int, ...] = (0, 1, 2, 3),
    eval_slice_rank: int = 0,
    boundary_kernel: int = 5,
    neighbor_kernel: int = 3,
    keep_ratio: float = 0.8,
    min_points: int = 10,
    bank_patient_indices: Tuple[int, ...] = (0, 1, 2, 3),
    bank_slice_ranks: Tuple[int, ...] = (0, 1),
    box_padding: int = 6,
    use_soft_uncertainty: bool = False,
) -> List[Dict[str, object]]:
    """在 PICAI 上可视化第三部分的 box 内双向 ordered core points。"""
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    se_source, seg_decoder = load_reference_source_models(device=device)

    bank = build_boundary_prototype_bank_from_reference_subset(
        device=device,
        se_source=se_source,
        seg_decoder=seg_decoder,
        patient_indices=bank_patient_indices,
        slice_ranks=bank_slice_ranks,
        boundary_kernel=boundary_kernel,
        neighbor_kernel=neighbor_kernel,
        keep_ratio=keep_ratio,
        min_points=min_points,
    )

    case_records: List[Dict[str, object]] = []
    for patient_index in eval_patient_indices:
        dataset = PICAITrainDataset(
            data_root=str(Path("/home/chenxu/dataset/picai").expanduser().resolve() / "train"),
            modality="t2w",
            n_slices=1,
        )
        slice_index = choose_slice_index_from_seg_volume(
            dataset.data_file["seg"][int(patient_index)],
            top_rank=eval_slice_rank,
        )
        case_record = build_ordered_core_visualization_case(
            device=device,
            se_source=se_source,
            seg_decoder=seg_decoder,
            bank=bank,
            patient_index=int(patient_index),
            slice_index=int(slice_index),
            boundary_kernel=boundary_kernel,
            neighbor_kernel=neighbor_kernel,
            box_padding=box_padding,
            use_soft_uncertainty=use_soft_uncertainty,
        )
        case_records.append(case_record)

    pseudo_path = save_dir / "picai_ordered_core_points_on_pseudo_labels.png"
    gt_path = save_dir / "picai_ordered_core_points_on_gt_labels.png"
    visualize_ordered_core_grid(
        case_records=case_records,
        label_key="pseudo_label",
        result_key="pseudo_core_results",
        save_path=pseudo_path,
    )
    visualize_ordered_core_grid(
        case_records=case_records,
        label_key="gt_label",
        result_key="gt_core_results",
        save_path=gt_path,
    )

    print("=" * 80)
    print("PICAI Ordered Core Point Visualization")
    print("=" * 80)
    print(f"device={device}")
    print(f"eval_patient_indices={eval_patient_indices}, eval_slice_rank={eval_slice_rank}")
    print(f"boundary_kernel={boundary_kernel}, neighbor_kernel={neighbor_kernel}, box_padding={box_padding}")
    print(f"use_soft_uncertainty={use_soft_uncertainty}")
    for case_record in case_records:
        pseudo_results = case_record["pseudo_core_results"]
        gt_results = case_record["gt_core_results"]
        print(
            f"patient={int(case_record['patient_index']):03d}, "
            f"slice={int(case_record['slice_index']):02d} | "
            f"pseudo_pairs={len(pseudo_results):2d} | "
            f"gt_pairs={len(gt_results):2d}"
        )
        for result_name, result_list in (("pseudo", pseudo_results), ("gt", gt_results)):
            for result in result_list:
                print(
                    f"  {result_name:>6s} pair={tuple(result['pair'])} | "
                    f"core_ab={result['core_ab']} | "
                    f"core_ba={result['core_ba']}"
                )
    print("-" * 80)
    print(f"pseudo-label core visualization saved to: {pseudo_path}")
    print(f"gt-label core visualization saved to:     {gt_path}")
    return case_records


def run_picai_pretrained_point_prompt_visualization(
    save_dir: str | Path,
    eval_patient_indices: Tuple[int, ...] = (0, 1, 2, 3),
    eval_slice_rank: int = 0,
    boundary_kernel: int = 5,
    neighbor_kernel: int = 3,
    keep_ratio: float = 0.8,
    min_points: int = 10,
    bank_patient_indices: Tuple[int, ...] = (0, 1, 2, 3),
    bank_slice_ranks: Tuple[int, ...] = (0, 1),
    box_padding: int = 6,
    use_soft_uncertainty: bool = False,
    offset_distance: float = 3.0,
    window_radius: int = 12,
    sigma: float = 5.0,
    topk_per_side: int = 3,
    min_distance: float = 5.0,
    min_score: Optional[float] = None,
) -> List[Dict[str, object]]:
    """在 PICAI 上可视化“第三部分核心点 -> 第四部分多点 prompts”的完整几何链路。"""
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    se_source, seg_decoder = load_reference_source_models(device=device)

    bank = build_boundary_prototype_bank_from_reference_subset(
        device=device,
        se_source=se_source,
        seg_decoder=seg_decoder,
        patient_indices=bank_patient_indices,
        slice_ranks=bank_slice_ranks,
        boundary_kernel=boundary_kernel,
        neighbor_kernel=neighbor_kernel,
        keep_ratio=keep_ratio,
        min_points=min_points,
    )

    case_records: List[Dict[str, object]] = []
    for patient_index in eval_patient_indices:
        dataset = PICAITrainDataset(
            data_root=str(Path("/home/chenxu/dataset/picai").expanduser().resolve() / "train"),
            modality="t2w",
            n_slices=1,
        )
        slice_index = choose_slice_index_from_seg_volume(
            dataset.data_file["seg"][int(patient_index)],
            top_rank=eval_slice_rank,
        )
        case_record = build_ordered_point_prompt_visualization_case(
            device=device,
            se_source=se_source,
            seg_decoder=seg_decoder,
            bank=bank,
            patient_index=int(patient_index),
            slice_index=int(slice_index),
            boundary_kernel=boundary_kernel,
            neighbor_kernel=neighbor_kernel,
            box_padding=box_padding,
            use_soft_uncertainty=use_soft_uncertainty,
            offset_distance=offset_distance,
            window_radius=window_radius,
            sigma=sigma,
            topk_per_side=topk_per_side,
            min_distance=min_distance,
            min_score=min_score,
        )
        case_records.append(case_record)

    pseudo_path = save_dir / "picai_point_prompts_on_pseudo_labels.png"
    gt_path = save_dir / "picai_point_prompts_on_gt_labels.png"
    visualize_point_prompt_grid(
        case_records=case_records,
        label_key="pseudo_label",
        result_key="pseudo_point_prompt_results",
        save_path=pseudo_path,
    )
    visualize_point_prompt_grid(
        case_records=case_records,
        label_key="gt_label",
        result_key="gt_point_prompt_results",
        save_path=gt_path,
    )

    print("=" * 80)
    print("PICAI Ordered Core -> Point Prompt Visualization")
    print("=" * 80)
    print(f"device={device}")
    print(f"eval_patient_indices={eval_patient_indices}, eval_slice_rank={eval_slice_rank}")
    print(
        f"boundary_kernel={boundary_kernel}, neighbor_kernel={neighbor_kernel}, "
        f"box_padding={box_padding}, offset_distance={offset_distance}, window_radius={window_radius}"
    )
    print(
        f"sigma={sigma}, topk_per_side={topk_per_side}, min_distance={min_distance}, "
        f"min_score={min_score}, use_soft_uncertainty={use_soft_uncertainty}"
    )
    for case_record in case_records:
        pseudo_results = case_record["pseudo_point_prompt_results"]
        gt_results = case_record["gt_point_prompt_results"]
        print(
            f"patient={int(case_record['patient_index']):03d}, "
            f"slice={int(case_record['slice_index']):02d} | "
            f"pseudo_pairs={len(pseudo_results):2d} | "
            f"gt_pairs={len(gt_results):2d}"
        )
        for result_name, result_list in (("pseudo", pseudo_results), ("gt", gt_results)):
            for result in result_list:
                print(
                    f"  {result_name:>6s} pair=({int(result['a'])},{int(result['b'])}) | "
                    f"ref_a=({float(result['ref_center_a']['y']):.1f},{float(result['ref_center_a']['x']):.1f}) | "
                    f"ref_b=({float(result['ref_center_b']['y']):.1f},{float(result['ref_center_b']['x']):.1f}) | "
                    f"points_a={len(result['points_a'])} | "
                    f"points_b={len(result['points_b'])}"
                )
    print("-" * 80)
    print(f"pseudo-label point-prompt visualization saved to: {pseudo_path}")
    print(f"gt-label point-prompt visualization saved to:     {gt_path}")
    return case_records


def main() -> None:
    """PICAI 预训练特征下的第三部分到第四部分联合可视化入口。"""
    run_picai_pretrained_point_prompt_visualization(
        save_dir=Path(__file__).resolve().parent / "outputs",
        eval_patient_indices=(0, 1, 2, 3),
        eval_slice_rank=0,
        boundary_kernel=5,
        neighbor_kernel=3,
        keep_ratio=0.8,
        min_points=10,
        bank_patient_indices=(0, 1, 2, 3),
        bank_slice_ranks=(0, 1),
        box_padding=6,
        use_soft_uncertainty=False,
        offset_distance=3.0,
        window_radius=12,
        sigma=5.0,
        topk_per_side=3,
        min_distance=5.0,
        min_score=None,
    )  # 默认先在几张真实 PICAI 切片上看 pseudo/gt 两张核心点几何总览图。


if __name__ == "__main__":  # 让脚本可直接执行。
    main()
