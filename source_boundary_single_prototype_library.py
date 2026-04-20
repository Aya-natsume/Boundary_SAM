from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from .source_fine_boundary_points import BoundaryKey, build_source_fine_boundary_points
except ImportError:
    from source_fine_boundary_points import BoundaryKey, build_source_fine_boundary_points


TensorDict = Dict[str, Any]


def canonicalize_boundary_key(a: int, b: int, ordered: bool = False) -> BoundaryKey:
    """
    将边界类别对统一规范化为稳定 key，避免同一条边界被重复统计。

    规范规则:
        1. `ordered=True` 时，严格保留输入顺序，返回 `(a, b)`
        2. `ordered=False` 时，器官-器官边界采用无序表示，统一为 `(min(a, b), max(a, b))`
        3. `ordered=False` 且存在背景类别 0 时，统一写成 `(foreground, 0)`，保持和第一步模块一致

    参数:
        a: int
            第一个类别编号。
        b: int
            第二个类别编号。
        ordered: bool
            是否保留有序边界表示。

    返回:
        boundary_key: Tuple[int, int]
            规范化后的边界 key。
    """
    a = int(a)
    b = int(b)
    if a == b:
        raise ValueError(f"边界类别对中的两个类别不能相同，当前为 {(a, b)}")

    if ordered:
        return a, b

    if a == 0 or b == 0:
        foreground_class = b if a == 0 else a
        return int(foreground_class), 0

    return tuple(sorted((a, b)))


def _ensure_feature_map_is_bchw(feature_map: torch.Tensor) -> torch.Tensor:
    """
    将输入特征图整理成 `(B, C, H, W)` 形式。

    参数:
        feature_map: torch.Tensor
            支持 `(C, H, W)` 或 `(B, C, H, W)`。

    返回:
        feature_map_bchw: torch.Tensor
            统一后的四维特征图。
    """
    if not isinstance(feature_map, torch.Tensor):
        raise TypeError("feature_map 必须是 torch.Tensor")

    # 单张图 `[C,H,W]` 的情况并不少见，这里补出 batch 维，后面接口就统一了。
    if feature_map.ndim == 3:
        feature_map = feature_map.unsqueeze(0)
    elif feature_map.ndim != 4:
        raise ValueError(f"feature_map 必须是 3/4 维张量，当前 shape={tuple(feature_map.shape)}")

    return feature_map


def _ensure_seg_label_is_bhw(seg_label: torch.Tensor) -> torch.Tensor:
    """
    将分割标签整理成 `(B, H, W)` 形式。

    参数:
        seg_label: torch.Tensor
            支持 `(H, W)` 或 `(B, H, W)`。

    返回:
        seg_label_bhw: torch.Tensor
            统一后的 long 标签张量。
    """
    if not isinstance(seg_label, torch.Tensor):
        raise TypeError("seg_label 必须是 torch.Tensor")

    # 单张标签图时自动补 batch 维，避免后续 batch 逻辑分叉得太难看。
    if seg_label.ndim == 2:
        seg_label = seg_label.unsqueeze(0)
    elif seg_label.ndim != 3:
        raise ValueError(f"seg_label 必须是 2/3 维张量，当前 shape={tuple(seg_label.shape)}")

    return seg_label.long()


def _ensure_coords_have_batch_index(
    coords: torch.Tensor,
    default_batch_idx: int = 0,
) -> torch.Tensor:
    """
    将点坐标统一整理为 `(N, 3)`，列顺序固定为 `(batch_idx, y, x)`。

    参数:
        coords: torch.Tensor
            支持 `(N, 2)` 或 `(N, 3)`。
        default_batch_idx: int
            当输入是 `(N, 2)` 时，自动补上的 batch 下标。

    返回:
        coords_batched: torch.Tensor
            带 batch 维索引的坐标张量。
    """
    if not isinstance(coords, torch.Tensor):
        raise TypeError("coords 必须是 torch.Tensor")
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError(f"coords 必须是 `(N, 2)` 或 `(N, 3)`，当前 shape={tuple(coords.shape)}")

    # 坐标索引必须是 long，不然高级索引会闹脾气。
    coords = coords.long()
    if coords.shape[1] == 3:
        return coords

    # 如果上游只给了 `(y,x)`，这里补一个默认 batch 下标，省得调用方额外包装。
    batch_column = torch.full(
        size=(coords.shape[0], 1),
        fill_value=int(default_batch_idx),
        device=coords.device,
        dtype=torch.long,
    )
    return torch.cat([batch_column, coords], dim=1)


def _gather_point_features(feature_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    从特征图中按坐标批量提取点特征。

    参数:
        feature_map: torch.Tensor
            特征图，形状为 `(B, C, H, W)`。
        coords: torch.Tensor
            坐标张量，形状为 `(N, 3)`，列顺序为 `(batch_idx, y, x)`。

    返回:
        point_features: torch.Tensor
            点特征张量，形状为 `(N, C)`。
    """
    feature_map = _ensure_feature_map_is_bchw(feature_map)
    coords = _ensure_coords_have_batch_index(coords)

    if coords.numel() == 0:
        return feature_map.new_zeros((0, feature_map.shape[1]))

    # 高级索引一次性把所有点特征拿出来，没必要回退到点级 Python 循环。
    batch_idx = coords[:, 0]
    y = coords[:, 1]
    x = coords[:, 2]
    return feature_map[batch_idx, :, y, x]


def _get_normalized_boundary_features(
    boundary_info: Mapping[str, Any],
    feature_map: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    获取边界点的 L2 归一化特征。

    参数:
        boundary_info: Mapping[str, Any]
            单个边界类别的字典信息。
        feature_map: Optional[torch.Tensor]
            当 `boundary_info` 未显式携带特征时，用于按坐标提取特征。

    返回:
        features_norm: torch.Tensor
            形状为 `(N, C)` 的归一化特征。
    """
    # 第一优先级直接复用上游已经算好的归一化特征，少做一次重复工作更稳。
    if "features_norm" in boundary_info:
        return boundary_info["features_norm"].to(dtype=torch.float32)

    # 如果只有原始特征，就在这里补做一次 L2 normalize。
    if "features" in boundary_info:
        features = boundary_info["features"].to(dtype=torch.float32)
        return F.normalize(features, p=2, dim=1, eps=1e-12)

    if feature_map is None:
        raise ValueError("boundary_info 未携带特征，且 feature_map=None，无法补取点特征")

    if "coords" not in boundary_info:
        raise KeyError("boundary_info 中缺少 coords，无法从 feature_map 中提取点特征")

    # 上游既没带原始特征也没带归一化特征，那就只能回到 feature_map 里按坐标补取。
    coords = _ensure_coords_have_batch_index(boundary_info["coords"])
    features = _gather_point_features(feature_map, coords).to(dtype=torch.float32)
    return F.normalize(features, p=2, dim=1, eps=1e-12)


def _extract_per_boundary_summary(summary_info: Optional[Mapping[Any, Any]]) -> Dict[BoundaryKey, Dict[str, Any]]:
    """
    从不同风格的 summary_info 中提取按边界类别组织的统计字典。

    参数:
        summary_info: Optional[Mapping[Any, Any]]
            可能是第一步模块的扁平字典，也可能是本模块 finalize 后的嵌套字典。

    返回:
        per_boundary_summary: Dict[BoundaryKey, Dict[str, Any]]
            统一后的边界统计字典。
    """
    if summary_info is None:
        return {}

    if not isinstance(summary_info, Mapping):
        raise TypeError("summary_info 必须是 Mapping 或 None")

    # 本模块自己的 finalize 输出会把统计放在 `per_boundary` 里；
    # 第一阶段模块则是扁平 dict。这里两种都接住，省得调用方来回判断。
    if "per_boundary" in summary_info and isinstance(summary_info["per_boundary"], Mapping):
        raw_mapping = summary_info["per_boundary"]
    else:
        raw_mapping = summary_info

    per_boundary_summary: Dict[BoundaryKey, Dict[str, Any]] = {}
    for key, value in raw_mapping.items():
        if not (isinstance(key, tuple) and len(key) == 2):
            continue
        if not isinstance(value, Mapping):
            continue
        canonical_key = canonicalize_boundary_key(key[0], key[1], ordered=False)
        per_boundary_summary[canonical_key] = dict(value)

    return per_boundary_summary


def compute_image_level_boundary_prototypes(
    boundary_dict: Dict[BoundaryKey, Dict[str, Any]],
    feature_map: torch.Tensor,
    ordered: bool = False,
) -> Dict[BoundaryKey, List[Dict[str, Any]]]:
    """
    基于第一步筛选后的高质量边界点，统计 batch 内每张图像的图级边界原型。

    统计规则严格遵守:
        1. 对当前图像当前边界类别的所有点特征先做 L2 normalize
        2. 在图像内对同类边界点特征求均值
        3. 图级 prototype 再做一次 L2 normalize

    参数:
        boundary_dict: Dict[BoundaryKey, Dict[str, Any]]
            第一步筛选后的边界点字典。
            常见字段包括：
            - `coords`: `(N, 3)` 或 `(N, 2)`
            - `features` 或 `features_norm`
            - `batch_idx`: `(N,)`，可选
        feature_map: torch.Tensor
            特征图，形状为 `(B, C, H, W)` 或 `(C, H, W)`。
            当 `boundary_dict` 未显式携带特征时，会从这里按坐标补取特征。
        ordered: bool
            是否保留有序边界表示。

    返回:
        image_proto_dict: Dict[BoundaryKey, List[Dict[str, Any]]]
            每个边界类别对应一个列表，列表内每个元素对应一张图像：
            {
                (A, B): [
                    {
                        "batch_idx": int,
                        "image_proto": Tensor[C],
                        "num_points": int
                    },
                    ...
                ]
            }
    """
    feature_map = _ensure_feature_map_is_bchw(feature_map)
    image_proto_dict: Dict[BoundaryKey, List[Dict[str, Any]]] = {}

    for pair_key, boundary_info in boundary_dict.items():
        # 所有统计都先走一次 key 规范化，避免 `(A,B)` / `(B,A)` 被重复累计。
        canonical_key = canonicalize_boundary_key(pair_key[0], pair_key[1], ordered=ordered)
        if "coords" not in boundary_info:
            raise KeyError(f"boundary_dict[{pair_key}] 缺少 coords 字段")

        coords = _ensure_coords_have_batch_index(boundary_info["coords"])
        if coords.numel() == 0:
            continue

        features_norm = _get_normalized_boundary_features(boundary_info, feature_map)
        if features_norm.ndim != 2:
            raise ValueError(
                f"boundary_dict[{pair_key}] 的点特征必须是二维张量，当前 shape={tuple(features_norm.shape)}"
            )
        if features_norm.shape[0] != coords.shape[0]:
            raise ValueError(
                f"boundary_dict[{pair_key}] 的点数和特征数不一致，"
                f"coords={coords.shape[0]}, features={features_norm.shape[0]}"
            )

        # 这里的统计单位必须是“图像”，所以先按 batch 内的图像下标拆开。
        batch_indices = coords[:, 0].unique(sorted=True)
        for batch_idx in batch_indices.tolist():
            image_mask = coords[:, 0].eq(int(batch_idx))
            num_points = int(image_mask.sum().item())
            if num_points <= 0:
                continue

            # 这里已经保证是“点先归一化”，然后才在图像内对该边界类别求均值。
            image_features_norm = features_norm[image_mask]
            # 图内均值只在当前图像、当前边界类别范围内计算，绝不跨图混点。
            image_proto = image_features_norm.mean(dim=0, keepdim=False)
            image_proto = F.normalize(
                image_proto.unsqueeze(0),
                p=2,
                dim=1,
                eps=1e-12,
            ).squeeze(0)

            image_proto_dict.setdefault(canonical_key, []).append(
                {
                    "batch_idx": int(batch_idx),
                    "image_proto": image_proto.to(dtype=torch.float32),
                    "num_points": num_points,
                }
            )

    return image_proto_dict


class SingleBoundaryPrototypeBuilder:
    """
    源域细粒度边界单原型库的流式累计器。

    设计目标:
        1. 只累计图级 prototype，不长期缓存所有点特征
        2. 用“两级统计”构建单原型：
           point normalize -> image mean -> image normalize -> global mean -> global normalize
        3. 当前阶段仅支持离线初始化，后续可以自然扩展为 EMA prototype bank

    参数:
        num_classes: int
            类别总数，包含背景 0。
        feature_dim: int
            特征通道维度 C。
        ordered: bool
            是否保留有序边界表示。
        device: Optional[Union[str, torch.device]]
            原型累加和所在设备。默认首次 update 时自动跟随输入原型设备。
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        ordered: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes 必须大于 1")
        if feature_dim <= 0:
            raise ValueError("feature_dim 必须大于 0")

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.ordered = bool(ordered)
        self.device = torch.device(device) if device is not None else None

        # 这里只维护流式累计量，不缓存点级特征，内存占用会干净很多。
        self.proto_sum: Dict[BoundaryKey, torch.Tensor] = {}
        self.img_count: Dict[BoundaryKey, int] = defaultdict(int)
        self.point_count: Dict[BoundaryKey, int] = defaultdict(int)
        self.raw_count: Dict[BoundaryKey, int] = defaultdict(int)
        self.filtered_count: Dict[BoundaryKey, int] = defaultdict(int)

    def _get_target_device(self, proto: torch.Tensor) -> torch.device:
        """
        获取累计器实际使用的设备。

        参数:
            proto: torch.Tensor
                当前输入的图级 prototype。

        返回:
            target_device: torch.device
                累计所使用的设备。
        """
        # 第一次 update 时再绑定设备，这样外面不传 device 也不会出奇怪分支。
        if self.device is None:
            self.device = proto.device
        return self.device

    def _ensure_boundary_slot(self, pair_key: BoundaryKey, proto: torch.Tensor) -> None:
        """
        确保指定边界类别已经初始化累计容器。

        参数:
            pair_key: BoundaryKey
                规范化后的边界 key。
            proto: torch.Tensor
                当前输入图级 prototype，用于决定 dtype/device。
        """
        if pair_key in self.proto_sum:
            return

        # 每个边界类别只初始化一次累计槽位，后续就直接原地累加。
        target_device = self._get_target_device(proto)
        self.proto_sum[pair_key] = torch.zeros(
            self.feature_dim,
            device=target_device,
            dtype=torch.float32,
        )

    def update(
        self,
        image_proto_dict: Dict[BoundaryKey, List[Dict[str, Any]]],
        batch_summary_info: Optional[Mapping[Any, Any]] = None,
    ) -> None:
        """
        使用当前 batch 的图级边界 prototype 更新全局累计器。

        参数:
            image_proto_dict: Dict[BoundaryKey, List[Dict[str, Any]]]
                `compute_image_level_boundary_prototypes` 的输出。
            batch_summary_info: Optional[Mapping[Any, Any]]
                可选的 batch 级统计信息。
                如果传入第一步模块的 `summary_info`，会自动累计 `raw_count/kept_count`。
        """
        for pair_key, image_proto_list in image_proto_dict.items():
            canonical_key = canonicalize_boundary_key(pair_key[0], pair_key[1], ordered=self.ordered)
            for image_info in image_proto_list:
                if "image_proto" not in image_info:
                    raise KeyError(f"image_proto_dict[{pair_key}] 中缺少 image_proto 字段")
                if "num_points" not in image_info:
                    raise KeyError(f"image_proto_dict[{pair_key}] 中缺少 num_points 字段")

                image_proto = image_info["image_proto"].to(dtype=torch.float32)
                if image_proto.ndim != 1 or image_proto.shape[0] != self.feature_dim:
                    raise ValueError(
                        f"图级 prototype 维度不匹配，期望 {(self.feature_dim,)}, 当前 shape={tuple(image_proto.shape)}"
                    )

                self._ensure_boundary_slot(canonical_key, image_proto)
                target_device = self.device if self.device is not None else image_proto.device
                # 第二级统计在这里完成：对所有图级 prototype 做简单平均所需的求和累计。
                self.proto_sum[canonical_key] += image_proto.to(device=target_device, dtype=torch.float32)
                self.img_count[canonical_key] += 1
                self.point_count[canonical_key] += int(image_info["num_points"])

        per_boundary_summary = _extract_per_boundary_summary(batch_summary_info)
        if per_boundary_summary:
            for pair_key, pair_summary in per_boundary_summary.items():
                canonical_key = canonicalize_boundary_key(pair_key[0], pair_key[1], ordered=self.ordered)
                self.raw_count[canonical_key] += int(pair_summary.get("raw_count", 0))
                self.filtered_count[canonical_key] += int(
                    pair_summary.get("kept_count", pair_summary.get("filtered_count", 0))
                )
        else:
            # 没有额外 summary 时，退化为把保留点数记成 filtered_count。
            for pair_key, image_proto_list in image_proto_dict.items():
                canonical_key = canonicalize_boundary_key(pair_key[0], pair_key[1], ordered=self.ordered)
                self.filtered_count[canonical_key] += int(
                    sum(int(image_info["num_points"]) for image_info in image_proto_list)
                )

    def finalize(self) -> Tuple[Dict[BoundaryKey, torch.Tensor], Dict[str, Any]]:
        """
        结束累计，生成最终的全局单原型库。

        全局统计规则:
            1. 对每个边界类别，将所有图级 prototype 求平均
            2. 对平均后的 global prototype 再做一次 L2 normalize

        返回:
            prototype_dict: Dict[BoundaryKey, torch.Tensor]
                每个边界类别对应一个 `(C,)` 的全局 prototype，默认放在 CPU 上。
            summary_info: Dict[str, Any]
                包含 `metadata` 和 `per_boundary` 两部分。
        """
        prototype_dict: Dict[BoundaryKey, torch.Tensor] = {}
        per_boundary_summary: Dict[BoundaryKey, Dict[str, Any]] = {}

        all_keys = sorted(
            set(self.proto_sum.keys())
            | set(self.img_count.keys())
            | set(self.point_count.keys())
            | set(self.raw_count.keys())
            | set(self.filtered_count.keys())
        )

        for pair_key in all_keys:
            img_count = int(self.img_count.get(pair_key, 0))
            point_count = int(self.point_count.get(pair_key, 0))
            raw_count = int(self.raw_count.get(pair_key, 0))
            filtered_count = int(self.filtered_count.get(pair_key, 0))

            if img_count > 0:
                # 先做跨图平均，再做最后一次 L2 normalize，这个顺序不能写反。
                global_proto = self.proto_sum[pair_key] / float(img_count)
                global_proto = F.normalize(
                    global_proto.unsqueeze(0),
                    p=2,
                    dim=1,
                    eps=1e-12,
                ).squeeze(0)
                global_proto = global_proto.detach().cpu().to(dtype=torch.float32)
                prototype_dict[pair_key] = global_proto
                prototype_shape = tuple(global_proto.shape)
                prototype_norm = float(global_proto.norm(p=2).item())
                valid = True
            else:
                prototype_shape = (self.feature_dim,)
                prototype_norm = 0.0
                valid = False

            per_boundary_summary[pair_key] = {
                "img_count": img_count,
                "point_count": point_count,
                "raw_count": raw_count,
                "filtered_count": filtered_count,
                "valid": valid,
                "prototype_shape": prototype_shape,
                "prototype_norm": prototype_norm,
            }

        summary_info = {
            "metadata": {
                "num_classes": self.num_classes,
                "feature_dim": self.feature_dim,
                "ordered": self.ordered,
                "version": "single_boundary_prototype_library_v1",
                "num_valid_boundaries": len(prototype_dict),
            },
            "per_boundary": per_boundary_summary,
        }
        return prototype_dict, summary_info


def _extract_feature_map_from_model_output(model_output: Any) -> torch.Tensor:
    """
    从不同风格的模型输出中解析出 feature_map。

    参数:
        model_output: Any
            模型前向输出。

    返回:
        feature_map: torch.Tensor
            解析出的 `(B, C, H, W)` 特征图。
    """
    if isinstance(model_output, torch.Tensor):
        return _ensure_feature_map_is_bchw(model_output)

    if isinstance(model_output, Mapping):
        # 这里故意兼容几种常见输出字段名，省得接不同骨干网络时到处写适配胶水。
        candidate_keys = [
            "feature_map",
            "features",
            "feat",
            "encoder_feature",
            "encoder_features",
            "encoder_feat",
        ]
        for key in candidate_keys:
            if key in model_output and isinstance(model_output[key], torch.Tensor):
                return _ensure_feature_map_is_bchw(model_output[key])
        raise KeyError(
            f"无法从模型输出字典中找到 feature_map，当前可用 key={list(model_output.keys())}"
        )

    if isinstance(model_output, (tuple, list)):
        for item in model_output:
            if isinstance(item, torch.Tensor) and item.ndim in (3, 4):
                return _ensure_feature_map_is_bchw(item)
        raise ValueError("模型输出是 tuple/list，但其中没有可识别的特征图张量")

    raise TypeError(f"不支持的模型输出类型: {type(model_output)}")


def _parse_boundary_point_output(boundary_output: Any) -> Tuple[Dict[BoundaryKey, Dict[str, Any]], Dict[str, Any]]:
    """
    解析边界点筛选函数的输出。

    参数:
        boundary_output: Any
            可能是第一步模块的完整结果字典，也可能直接就是筛选后的 boundary_dict。

    返回:
        filtered_boundary_dict: Dict[BoundaryKey, Dict[str, Any]]
            筛选后的边界点字典。
        batch_summary_info: Dict[str, Any]
            对应的统计信息字典。
        """
    if not isinstance(boundary_output, Mapping):
        raise TypeError("boundary_point_fn 的输出必须是 dict 或 Mapping")

    # 第一阶段标准输出是完整结果 dict；如果调用方自己做了包装，这里也允许直接传筛选后字典。
    if "filtered_boundary_dict" in boundary_output:
        filtered_boundary_dict = boundary_output["filtered_boundary_dict"]
        batch_summary_info = dict(boundary_output.get("summary_info", {}))
        return filtered_boundary_dict, batch_summary_info

    return dict(boundary_output), {}


def _unpack_dataloader_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 dataloader 的 batch 中解析图像和标签。

    参数:
        batch: Any
            支持常见的 dict / tuple / list 风格 batch。

    返回:
        image: torch.Tensor
            输入图像，形状通常为 `(B, C_in, H, W)`。
        seg_label: torch.Tensor
            分割标签，形状为 `(B, H, W)`。
    """
    if isinstance(batch, Mapping):
        # 这里把常见 key 名都兜一下，尽量少给现有 dataloader 添麻烦。
        image_keys = ["image", "img", "input", "inputs", "data", "x"]
        label_keys = ["seg_label", "label", "mask", "target", "y"]

        image = None
        seg_label = None
        for key in image_keys:
            if key in batch:
                image = batch[key]
                break
        for key in label_keys:
            if key in batch:
                seg_label = batch[key]
                break

        if image is None or seg_label is None:
            raise KeyError(
                f"无法从 batch 字典中解析 image/seg_label，当前可用 key={list(batch.keys())}"
            )
        return image, seg_label

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise TypeError("dataloader batch 必须是 dict，或至少包含 `(image, seg_label)` 的 tuple/list")


def build_source_boundary_prototype_library(
    dataloader: Iterable[Any],
    model: nn.Module,
    num_classes: int,
    device: Union[str, torch.device],
    boundary_point_fn: Callable[..., Any],
    save_path: Optional[Union[str, Path]] = None,
    ordered: bool = False,
) -> Tuple[Dict[BoundaryKey, torch.Tensor], Dict[str, Any]]:
    """
    离线扫描整个源域数据集，构建细粒度边界单原型库。

    处理流程:
        1. 遍历 dataloader 一遍
        2. 前向得到 feature_map
        3. 调用第一步边界点筛选函数得到 `filtered_boundary_dict`
        4. 计算 batch 内图级 prototype
        5. 使用流式累计器更新全局统计
        6. 遍历结束后 finalize，得到全局单原型库

    参数:
        dataloader: Iterable[Any]
            源域 dataloader。
        model: nn.Module
            特征提取模型。输出应能解析出 `(B, C, H, W)` 的特征图。
        num_classes: int
            类别总数，包含背景 0。
        device: Union[str, torch.device]
            模型和输入所使用的设备。
        boundary_point_fn: Callable[..., Any]
            第一步边界点筛选函数，推荐直接传 `build_source_fine_boundary_points` 或其包装函数。
            调用方式约定为：
            `boundary_point_fn(feature_map=..., seg_label=..., num_classes=...)`
        save_path: Optional[Union[str, Path]]
            若不为 None，则自动保存原型库到磁盘。
        ordered: bool
            是否保留有序边界表示。

    返回:
        prototype_dict: Dict[BoundaryKey, torch.Tensor]
            全局单原型库。
        summary_info: Dict[str, Any]
            原型统计信息和 metadata。
    """
    device = torch.device(device)
    # 离线初始化阶段不该让 BN / Dropout 乱飘，所以先切到 eval，结束后再恢复。
    model_was_training = model.training
    model = model.to(device)
    model.eval()

    builder: Optional[SingleBoundaryPrototypeBuilder] = None

    with torch.no_grad():
        # 整个流程只扫一遍 dataloader，符合“离线初始化一次”的设计要求。
        for batch in dataloader:
            image, seg_label = _unpack_dataloader_batch(batch)
            image = image.to(device=device, non_blocking=True)
            seg_label = _ensure_seg_label_is_bhw(seg_label).to(device=device, non_blocking=True)

            # 模型输出可以是 tensor / dict / tuple，这里统一抽成 feature_map。
            model_output = model(image)
            feature_map = _extract_feature_map_from_model_output(model_output).to(dtype=torch.float32)

            if feature_map.shape[0] != seg_label.shape[0]:
                raise ValueError(
                    f"feature_map 与 seg_label 的 batch 大小不一致，"
                    f"feature_map.shape[0]={feature_map.shape[0]}, seg_label.shape[0]={seg_label.shape[0]}"
                )

            if feature_map.shape[-2:] != seg_label.shape[-2:]:
                raise ValueError(
                    f"feature_map 与 seg_label 的空间尺寸必须一致，"
                    f"当前 feature_map.shape[-2:]={tuple(feature_map.shape[-2:])}, "
                    f"seg_label.shape[-2:]={tuple(seg_label.shape[-2:])}"
                )

            # 第一阶段先产出高质量边界点，第二阶段再在这些点上做图级原型统计。
            boundary_output = boundary_point_fn(
                feature_map=feature_map,
                seg_label=seg_label,
                num_classes=num_classes,
            )
            filtered_boundary_dict, batch_summary_info = _parse_boundary_point_output(boundary_output)
            # 这一层只负责图像内统计，不做跨 batch / 跨数据集平均。
            image_proto_dict = compute_image_level_boundary_prototypes(
                boundary_dict=filtered_boundary_dict,
                feature_map=feature_map,
                ordered=ordered,
            )

            # feature_dim 以第一批真实特征为准初始化，避免外面再手工同步一份配置。
            if builder is None:
                builder = SingleBoundaryPrototypeBuilder(
                    num_classes=num_classes,
                    feature_dim=feature_map.shape[1],
                    ordered=ordered,
                    device=feature_map.device,
                )

            builder.update(
                image_proto_dict=image_proto_dict,
                batch_summary_info=batch_summary_info,
            )

    if model_was_training:
        model.train()

    if builder is None:
        raise ValueError("dataloader 为空，无法构建原型库")

    prototype_dict, summary_info = builder.finalize()

    # 保存是可选项，默认返回内存中的原型库；需要落盘时再显式传路径。
    if save_path is not None:
        save_boundary_prototype_library(
            prototype_dict=prototype_dict,
            summary_info=summary_info,
            save_path=save_path,
        )

    return prototype_dict, summary_info


def save_boundary_prototype_library(
    prototype_dict: Dict[BoundaryKey, torch.Tensor],
    summary_info: Dict[str, Any],
    save_path: Union[str, Path],
) -> None:
    """
    将边界单原型库保存到磁盘。

    保存内容至少包括:
        - prototype_dict
        - summary_info
        - num_classes
        - feature_dim
        - ordered
        - version

    参数:
        prototype_dict: Dict[BoundaryKey, torch.Tensor]
            待保存的原型字典。
        summary_info: Dict[str, Any]
            `SingleBoundaryPrototypeBuilder.finalize()` 返回的统计信息。
        save_path: Union[str, Path]
            保存路径，建议使用 `.pt`。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # metadata 缺字段时在这里补齐，保证磁盘文件结构尽量自描述。
    metadata = dict(summary_info.get("metadata", {}))
    if "num_classes" not in metadata:
        metadata["num_classes"] = None
    if "feature_dim" not in metadata:
        if prototype_dict:
            first_proto = next(iter(prototype_dict.values()))
            metadata["feature_dim"] = int(first_proto.shape[0])
        else:
            metadata["feature_dim"] = None
    if "ordered" not in metadata:
        metadata["ordered"] = None
    if "version" not in metadata:
        metadata["version"] = "single_boundary_prototype_library_v1"

    payload = {
        # prototype 保存到 CPU，加载时更稳，也不会把无关显存状态写进文件里。
        "prototype_dict": {
            pair_key: proto.detach().cpu().to(dtype=torch.float32)
            for pair_key, proto in prototype_dict.items()
        },
        "summary_info": summary_info,
        "num_classes": metadata["num_classes"],
        "feature_dim": metadata["feature_dim"],
        "ordered": metadata["ordered"],
        "version": metadata["version"],
    }
    torch.save(payload, save_path)


def load_boundary_prototype_library(
    load_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """
    从磁盘加载边界单原型库。

    参数:
        load_path: Union[str, Path]
            原型库文件路径。
        map_location: Union[str, torch.device]
            `torch.load` 的加载设备。

    返回:
        payload: Dict[str, Any]
            包含：
            - `prototype_dict`
            - `summary_info`
            - `num_classes`
            - `feature_dim`
            - `ordered`
            - `version`
    """
    load_path = Path(load_path)
    # 新版 torch 支持 `weights_only=False`，老版本不支持，所以这里做一个兼容兜底。
    try:
        payload = torch.load(load_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(load_path, map_location=map_location)

    required_keys = ["prototype_dict", "summary_info"]
    for key in required_keys:
        if key not in payload:
            raise KeyError(f"加载的原型库文件缺少必要字段 `{key}`")

    return payload


def inspect_boundary_prototype_library(
    prototype_dict: Dict[BoundaryKey, torch.Tensor],
    summary_info: Dict[str, Any],
    min_img_count: int = 2,
    min_point_count: int = 20,
) -> List[Dict[str, Any]]:
    """
    打印边界单原型库的统计信息，便于快速判断哪些边界类别可靠。

    打印内容包括:
        - 边界类别
        - img_count
        - point_count
        - raw_count
        - filtered_count
        - prototype 维度
        - prototype 向量范数
        - 是否低样本警告

    参数:
        prototype_dict: Dict[BoundaryKey, torch.Tensor]
            原型字典。
        summary_info: Dict[str, Any]
            对应统计信息。
        min_img_count: int
            图像数低于该值时给出提醒。
        min_point_count: int
            点数低于该值时给出提醒。

    返回:
        inspection_records: List[Dict[str, Any]]
            结构化检查结果，方便后续程序化处理。
    """
    # inspect 本质上只是个诊断工具，所以一边打印一边也返回结构化结果，后面好复用。
    per_boundary = _extract_per_boundary_summary(summary_info)
    metadata = dict(summary_info.get("metadata", {}))

    print("=" * 100)
    print("源域细粒度边界单原型库检查")
    print("=" * 100)
    print(
        f"metadata: num_classes={metadata.get('num_classes')}, "
        f"feature_dim={metadata.get('feature_dim')}, "
        f"ordered={metadata.get('ordered')}, "
        f"num_valid_boundaries={metadata.get('num_valid_boundaries')}"
    )

    if not per_boundary:
        print("没有任何边界类别统计信息。")
        return []

    inspection_records: List[Dict[str, Any]] = []
    for pair_key in sorted(per_boundary.keys()):
        info = per_boundary[pair_key]
        proto = prototype_dict.get(pair_key, None)
        prototype_dim = tuple(proto.shape) if proto is not None else tuple(info.get("prototype_shape", ()))
        prototype_norm = float(proto.norm(p=2).item()) if proto is not None else float(info.get("prototype_norm", 0.0))
        img_count = int(info.get("img_count", 0))
        point_count = int(info.get("point_count", 0))
        raw_count = int(info.get("raw_count", 0))
        filtered_count = int(info.get("filtered_count", 0))

        warnings: List[str] = []
        if img_count < min_img_count:
            warnings.append(f"img_count<{min_img_count}")
        if point_count < min_point_count:
            warnings.append(f"point_count<{min_point_count}")
        if proto is None:
            warnings.append("prototype_missing")

        warning_text = " | ".join(warnings) if warnings else "OK"
        print(
            f"boundary={pair_key}, "
            f"img_count={img_count}, "
            f"point_count={point_count}, "
            f"raw_count={raw_count}, "
            f"filtered_count={filtered_count}, "
            f"proto_dim={prototype_dim}, "
            f"proto_norm={prototype_norm:.6f}, "
            f"status={warning_text}"
        )

        inspection_records.append(
            {
                "pair_key": pair_key,
                "img_count": img_count,
                "point_count": point_count,
                "raw_count": raw_count,
                "filtered_count": filtered_count,
                "prototype_dim": prototype_dim,
                "prototype_norm": prototype_norm,
                "warnings": warnings,
            }
        )

    return inspection_records


class DummyBoundaryDataset(Dataset):
    """
    用于最小可运行 demo 的二维假数据集。

    数据内容:
        - `image`: `(1, H, W)` 的单通道图像
        - `seg_label`: `(H, W)` 的整型标签图

    边界结构设计:
        1. 类别 1 和类别 2 在上半区域直接接壤，稳定产生 `(1, 2)` 边界
        2. 类别 3 位于下半区域，与背景接壤，稳定产生 `(3, 0)` 边界
        3. 类别 1 和背景、类别 2 和背景也会自然出现
    """

    def __init__(
        self,
        num_samples: int = 6,
        height: int = 96,
        width: int = 96,
    ) -> None:
        self.num_samples = int(num_samples)
        self.height = int(height)
        self.width = int(width)

    def __len__(self) -> int:
        """
        返回数据集长度。
        """
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        返回单个样本。

        参数:
            index: int
                样本索引。

        返回:
            sample: Dict[str, torch.Tensor]
                包含 `image` 和 `seg_label`。
        """
        seg_label = self._create_seg_label(index=index)
        image = self._create_image_from_label(seg_label=seg_label, index=index)
        return {
            "image": image,
            "seg_label": seg_label,
        }

    def _create_seg_label(self, index: int) -> torch.Tensor:
        """
        生成带有轻微形变和位移的标签图。

        参数:
            index: int
                样本索引。

        返回:
            seg_label: torch.Tensor
                形状为 `(H, W)` 的 long 标签图。
        """
        height = self.height
        width = self.width
        seg_label = torch.zeros((height, width), dtype=torch.long)

        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )

        # 给每个样本一点轻微位移，免得所有图级 prototype 都像复制粘贴出来的。
        shift_y = (index % 3) - 1
        shift_x = ((index // 3) % 3) - 1

        organ1_y1 = 14 + 2 * shift_y
        organ1_y2 = 56 + 2 * shift_y
        organ1_x1 = 14 + 2 * shift_x
        organ1_x2 = 36 + 2 * shift_x
        organ1_mask = (
            (yy >= organ1_y1)
            & (yy <= organ1_y2)
            & (xx >= organ1_x1)
            & (xx <= organ1_x2)
        )
        seg_label[organ1_mask] = 1

        organ2_y1 = 18 + shift_y
        organ2_y2 = 60 + shift_y
        organ2_x1 = organ1_x2 + 1
        organ2_x2 = organ2_x1 + 26
        organ2_mask = (
            (yy >= organ2_y1)
            & (yy <= organ2_y2)
            & (xx >= organ2_x1)
            & (xx <= organ2_x2)
        )
        seg_label[organ2_mask] = 2

        ellipse_center_y = 74 + 2 * shift_y
        ellipse_center_x = 34 + 3 * shift_x
        ellipse_mask = (
            ((yy - float(ellipse_center_y)) / 14.0) ** 2
            + ((xx - float(ellipse_center_x)) / 20.0) ** 2
        ) <= 1.0
        seg_label[ellipse_mask] = 3

        return seg_label

    def _create_image_from_label(self, seg_label: torch.Tensor, index: int) -> torch.Tensor:
        """
        根据标签图生成一个简单的单通道假图像。

        参数:
            seg_label: torch.Tensor
                形状为 `(H, W)` 的标签图。
            index: int
                样本索引，用于控制噪声随机性。

        返回:
            image: torch.Tensor
                形状为 `(1, H, W)` 的浮点图像。
        """
        height, width = seg_label.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, steps=height),
            torch.linspace(0.0, 1.0, steps=width),
            indexing="ij",
        )

        label_float = seg_label.to(dtype=torch.float32)
        image = 0.15 * yy + 0.10 * xx
        image = image + 0.25 * (label_float == 1).to(torch.float32)
        image = image + 0.45 * (label_float == 2).to(torch.float32)
        image = image + 0.65 * (label_float == 3).to(torch.float32)

        # 噪声种子跟 index 绑定，既保留样本差异，也保证 demo 复现稳定。
        generator = torch.Generator()
        generator.manual_seed(2026 + int(index))
        noise = 0.03 * torch.randn((height, width), generator=generator, dtype=torch.float32)
        image = image + noise

        return image.unsqueeze(0)


class DummyBoundaryFeatureModel(nn.Module):
    """
    用于 demo 的轻量特征提取模型。

    输入输出:
        - 输入 `image`: `(B, 1, H, W)`
        - 输出 `feature_map`: `(B, C, H, W)`

    设计思路:
        1. 用卷积从图像中提取一部分局部纹理特征
        2. 再拼接一组显式坐标特征，保证空间结构稳定
        3. 最终输出供第一步边界点筛选和第二步原型统计复用
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 12) -> None:
        super().__init__()
        if feature_dim < 4:
            raise ValueError("demo 模型的 feature_dim 至少应为 4")

        # 留出 4 个通道给显式坐标特征，剩下的维度交给卷积自己学。
        learned_dim = feature_dim - 4
        self.feature_dim = int(feature_dim)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, learned_dim, kernel_size=3, padding=1),
        )

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向计算特征图。

        参数:
            image: torch.Tensor
                输入图像，形状为 `(B, 1, H, W)`。

        返回:
            output: Dict[str, torch.Tensor]
                包含 `feature_map` 字段。
        """
        if image.ndim != 4:
            raise ValueError(f"image 必须是 `(B, C, H, W)`，当前 shape={tuple(image.shape)}")

        batch_size, _, height, width = image.shape
        learned_feature = self.backbone(image)

        # 显式坐标特征能让 demo 里的边界结构更稳定，不至于完全依赖随机卷积初始化。
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, steps=height, device=image.device, dtype=image.dtype),
            torch.linspace(0.0, 1.0, steps=width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        coord_feature = torch.stack(
            [
                yy,
                xx,
                torch.sin(math.pi * yy),
                torch.cos(math.pi * xx),
            ],
            dim=0,
        ).unsqueeze(0).expand(batch_size, -1, -1, -1)

        feature_map = torch.cat([learned_feature, coord_feature], dim=1)
        if feature_map.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"dummy model 输出通道数不正确，期望 {self.feature_dim}，当前 {feature_map.shape[1]}"
            )

        return {"feature_map": feature_map}


def _build_demo_boundary_point_fn() -> Callable[..., Dict[str, Any]]:
    """
    构造 demo 使用的第一步边界点筛选函数包装器。

    返回:
        boundary_point_fn: Callable[..., Dict[str, Any]]
            已固定筛选超参数的可调用对象。
    """

    def boundary_point_fn(feature_map: torch.Tensor, seg_label: torch.Tensor, num_classes: int) -> Dict[str, Any]:
        return build_source_fine_boundary_points(
            feature_map=feature_map,
            seg_label=seg_label,
            num_classes=num_classes,
            boundary_kernel=3,
            keep_ratio=0.8,
            min_points=8,
            ignore_background_order=True,
        )

    return boundary_point_fn


def run_minimal_demo() -> Dict[str, Any]:
    """
    运行最小可执行 demo，验证离线单原型库构建流程完整可用。

    demo 步骤:
        1. 构造 dummy dataset / dataloader
        2. 构造 dummy model
        3. 复用第一步边界点筛选函数
        4. 构建离线边界单原型库
        5. 打印统计信息
        6. 保存并重新加载，确认结构无误

    返回:
        demo_result: Dict[str, Any]
            包含 prototype_dict、summary_info、loaded_payload、save_path。
    """
    # 固定随机种子，避免 demo 每次跑出来的统计都飘来飘去，不利于检查。
    torch.manual_seed(2026)

    dataset = DummyBoundaryDataset(num_samples=6, height=96, width=96)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    model = DummyBoundaryFeatureModel(in_channels=1, feature_dim=12)
    boundary_point_fn = _build_demo_boundary_point_fn()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = Path(__file__).resolve().parent / "demo_boundary_single_prototype_library.pt"

    prototype_dict, summary_info = build_source_boundary_prototype_library(
        dataloader=dataloader,
        model=model,
        num_classes=4,
        device=device,
        boundary_point_fn=boundary_point_fn,
        save_path=save_path,
        ordered=False,
    )

    inspect_boundary_prototype_library(
        prototype_dict=prototype_dict,
        summary_info=summary_info,
        min_img_count=2,
        min_point_count=20,
    )

    # 重新加载一遍，确保磁盘保存结构和后续直接查询 prototype 的用法是通的。
    loaded_payload = load_boundary_prototype_library(save_path, map_location="cpu")
    loaded_proto_dict = loaded_payload["prototype_dict"]
    print("=" * 100)
    print("重新加载后的原型库检查")
    print("=" * 100)
    print(
        f"loaded metadata: num_classes={loaded_payload.get('num_classes')}, "
        f"feature_dim={loaded_payload.get('feature_dim')}, "
        f"ordered={loaded_payload.get('ordered')}, "
        f"num_prototypes={len(loaded_proto_dict)}"
    )
    for pair_key in sorted(loaded_proto_dict.keys()):
        proto = loaded_proto_dict[pair_key]
        print(
            f"loaded boundary={pair_key}, "
            f"shape={tuple(proto.shape)}, "
            f"norm={float(proto.norm(p=2).item()):.6f}"
        )

    return {
        "prototype_dict": prototype_dict,
        "summary_info": summary_info,
        "loaded_payload": loaded_payload,
        "save_path": str(save_path),
    }


if __name__ == "__main__":
    run_minimal_demo()
