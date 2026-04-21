"""源域细粒度边界动态 prototype bank 模块。

本文件只实现第二部分：
1. 基于第一部分输出的高质量边界点与点特征，计算图级 boundary prototype。
2. 使用“动态字典 + EMA 更新”维护细粒度边界 prototype bank。
3. 支持 bank 的检查、保存、加载与最小可运行 demo。

注意：
1. 当前文件故意不实现目标域 score、SAM prompt、SAM refinement、多子原型与离线全量统计。
2. 当前文件只做“动态图 prototype bank + EMA 更新”，先把主干逻辑做扎实。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


BoundaryKey = Tuple[int, int]
ImagePrototypeDict = Dict[BoundaryKey, List[Dict[str, torch.Tensor | int]]]
BankSummaryDict = Dict[BoundaryKey, Dict[str, int | float | Tuple[int, ...]]]


def canonicalize_ordered_boundary_key(a: int, b: int) -> BoundaryKey:
    """规范化有序细粒度边界类别 key。

    输入：
        a: int
            当前边界点所属类别编号。
        b: int
            当前边界点邻域接触到的类别编号。

    输出：
        boundary_key: Tuple[int, int]
            规范化后的有序边界 key，固定保持为 (a, b)。

    说明：
        1. 当前项目现在只允许 ordered boundary。
        2. (A, B) 与 (B, A) 是两个不同类别，bank 中必须分别维护两个 prototype。
    """
    return int(a), int(b)


def _l2_normalize_vector(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """对单个向量做 L2 normalize。"""
    if vector.dim() != 1:
        raise ValueError("vector must have shape (C,)")
    # 这里单独封成 helper，不只是为了少写一行；
    # 更重要的是 bank 更新、加载和图级原型计算都要共用同一套归一化约定。
    return F.normalize(vector.unsqueeze(0), p=2, dim=1, eps=eps).squeeze(0)


def compute_image_level_boundary_prototypes(
    filtered_boundary_dict: Dict[BoundaryKey, Dict[str, torch.Tensor | int | float | bool]],
    feature_already_normalized: bool = True,
) -> ImagePrototypeDict:
    """按“图像”为单位计算每个细粒度边界类别的图级 prototype。

    输入：
        filtered_boundary_dict: dict
            第一部分输出的高质量边界点字典。
            建议至少包含：
            {
                (A, B): {
                    "coords": Tensor[N, 3],
                    "features": Tensor[N, C],
                    ...
                }
            }
            其中：
            - `coords` 的坐标顺序是 `[batch_idx, y, x]`
            - `features` 是当前边界类别对应的点特征
        feature_already_normalized: bool
            点特征是否已经做过 L2 normalize。
            - True: 不再对点特征重复归一化。
            - False: 会先对点特征做一次 L2 normalize。
    输出：
        image_proto_dict: dict
            形式示例：
            {
                (A, B): [
                    {
                        "batch_idx": int,
                        "image_proto": Tensor[C],
                        "num_points": int,
                    },
                    ...
                ]
            }

    说明：
        1. 这里的统计单位是“图像”，不是整个 batch 直接混合。
        2. 点特征如果已经归一化，就不要重复归一化；但图级 prototype 在求均值后仍然必须再归一化。
    """
    image_proto_dict: ImagePrototypeDict = {}

    for raw_key, boundary_data in filtered_boundary_dict.items():
        # 这里每个 ordered key 必须独立处理；
        # 一旦把 `(A,B)` 与 `(B,A)` 混合，后面第三步沿法线两侧做相似度扫描就会失去方向性。
        boundary_key = canonicalize_ordered_boundary_key(raw_key[0], raw_key[1])
        coords = boundary_data.get("coords")
        features = boundary_data.get("features")

        if not isinstance(coords, torch.Tensor):
            raise TypeError(f"filtered_boundary_dict[{boundary_key}]['coords'] must be a torch.Tensor")
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"filtered_boundary_dict[{boundary_key}]['features'] must be a torch.Tensor")
        if coords.dim() != 2 or coords.size(1) != 3:
            raise ValueError(f"coords for boundary {boundary_key} must have shape (N, 3)")
        if features.dim() != 2:
            raise ValueError(f"features for boundary {boundary_key} must have shape (N, C)")
        if coords.size(0) != features.size(0):
            raise ValueError(f"coords and features for boundary {boundary_key} must share the same first dimension")
        if coords.numel() == 0:
            continue

        batch_indices = coords[:, 0].long()
        unique_batch_indices = torch.unique(batch_indices)
        image_proto_entries: List[Dict[str, torch.Tensor | int]] = []

        for batch_idx in unique_batch_indices.tolist():
            image_mask = batch_indices == int(batch_idx)
            image_features = features[image_mask]
            if image_features.size(0) == 0:
                continue

            if not feature_already_normalized:
                # 这里只在必要时补一次点级归一化，避免外部已经归一化过时重复改尺度。
                image_features = F.normalize(image_features, p=2, dim=1, eps=1e-12)

            # 图级 prototype 这里仍然用简单均值，是因为当前 bank 只承担“主方向锚定”角色；
            # 在没有明确多原型需求前，先别把第二部分复杂化。
            image_proto = image_features.mean(dim=0)
            image_proto = _l2_normalize_vector(image_proto)

            image_proto_entries.append(
                {
                    "batch_idx": int(batch_idx),
                    "image_proto": image_proto,
                    "num_points": int(image_features.size(0)),
                }
            )

        if len(image_proto_entries) > 0:
            image_proto_dict[boundary_key] = image_proto_entries

    return image_proto_dict


class DynamicBoundaryPrototypeBank:
    """动态细粒度边界 prototype bank。

    功能：
        1. bank 内部用动态字典存储，不预枚举所有边界类别。
        2. 新边界类别首次出现时自动注册。
        3. 已存在边界类别使用 EMA 方式更新。
        4. 每次更新后都对 bank prototype 做一次 L2 normalize，便于后续余弦相似度匹配。
    """

    def __init__(
        self,
        feature_dim: int,
        momentum: float = 0.9,
        device: Optional[str | torch.device] = None,
    ) -> None:
        """初始化动态 prototype bank。

        输入：
            feature_dim: int
                prototype 特征维度 C。
            momentum: float
                EMA 动量系数。
            device: Optional[str | torch.device]
                bank 内部 prototype 所在设备。
        """
        if feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must be in the range [0, 1)")

        # bank 的 feature_dim 一旦确定，后面所有加载和更新都按这个维度校验；
        # 如果这里放松检查，错维度权重会在更后面的阶段才炸，排查会很麻烦。
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.ordered = True
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.bank: Dict[BoundaryKey, Dict[str, torch.Tensor | int]] = {}

    def _normalize_key(self, a: int, b: int) -> BoundaryKey:
        """内部统一做 key 规范化。"""
        return canonicalize_ordered_boundary_key(a, b)

    def update_from_image_prototypes(self, image_proto_dict: ImagePrototypeDict) -> Dict[BoundaryKey, Dict[str, int | float]]:
        """使用图级 prototype 更新 bank。

        输入：
            image_proto_dict: dict
                compute_image_level_boundary_prototypes 的输出。

        输出：
            update_info: dict
                当前这次更新的简要统计信息，便于外部记录日志。
        """
        update_info: Dict[BoundaryKey, Dict[str, int | float]] = {}

        for raw_key, image_entries in image_proto_dict.items():
            boundary_key = self._normalize_key(raw_key[0], raw_key[1])
            for entry in image_entries:
                image_proto = entry["image_proto"]
                batch_idx = entry["batch_idx"]
                num_points = entry["num_points"]

                if not isinstance(image_proto, torch.Tensor):
                    raise TypeError(f"image_proto for boundary {boundary_key} must be a torch.Tensor")
                if image_proto.dim() != 1 or image_proto.numel() != self.feature_dim:
                    raise ValueError(
                        f"image_proto for boundary {boundary_key} must have shape ({self.feature_dim},)"
                    )

                # 更新前先把图级 prototype 拉到 bank 所在设备上，避免 EMA 时设备不一致。
                image_proto = image_proto.to(self.device, dtype=torch.float32)
                image_proto = _l2_normalize_vector(image_proto)

                if boundary_key not in self.bank:
                    # 新 key 首次出现时直接注册，不做冷启动平均；
                    # 边界类本来就稀疏，首帧如果再被过度平滑，方向信息会更弱。
                    bank_prototype = image_proto.clone()
                    update_count = 1
                    point_count_total = int(num_points)
                else:
                    old_prototype = self.bank[boundary_key]["prototype"]
                    if not isinstance(old_prototype, torch.Tensor):
                        raise TypeError(f"stored prototype for boundary {boundary_key} must be a torch.Tensor")
                    # 这里保持标准 EMA 更新，原因是 bank 要兼顾稳定和新样本适应；
                    # 若直接改成简单均值，早期 noisy batch 会更容易长期污染 prototype。
                    bank_prototype = self.momentum * old_prototype + (1.0 - self.momentum) * image_proto
                    bank_prototype = _l2_normalize_vector(bank_prototype)
                    update_count = int(self.bank[boundary_key]["update_count"]) + 1
                    point_count_total = int(self.bank[boundary_key]["point_count_total"]) + int(num_points)

                self.bank[boundary_key] = {
                    "prototype": bank_prototype,
                    "update_count": update_count,
                    "point_count_total": point_count_total,
                    "last_batch_idx": int(batch_idx),
                }

                update_info[boundary_key] = {
                    "update_count": update_count,
                    "point_count_total": point_count_total,
                    "prototype_norm": float(bank_prototype.norm(p=2).item()),
                }

        return update_info

    def get(self, a: int, b: int) -> Optional[torch.Tensor]:
        """根据 (a, b) 获取当前 bank 中的 prototype。

        输入：
            a: int
                第一个类别编号。
            b: int
                第二个类别编号。

        输出：
            prototype: Optional[Tensor]
                如果存在则返回 Tensor[C]，否则返回 None。
        """
        # 访问时也统一走 ordered key 规范化，避免调用方传入 tuple 顺序不一致。
        boundary_key = self._normalize_key(a, b)
        if boundary_key not in self.bank:
            return None
        prototype = self.bank[boundary_key]["prototype"]
        if not isinstance(prototype, torch.Tensor):
            raise TypeError(f"stored prototype for boundary {boundary_key} must be a torch.Tensor")
        return prototype

    def state_dict(self) -> Dict[str, object]:
        """返回可保存的 bank 状态字典。"""
        bank_state: Dict[BoundaryKey, Dict[str, torch.Tensor | int]] = {}
        for boundary_key, entry in self.bank.items():
            prototype = entry["prototype"]
            if not isinstance(prototype, torch.Tensor):
                raise TypeError(f"stored prototype for boundary {boundary_key} must be a torch.Tensor")
            bank_state[boundary_key] = {
                "prototype": prototype.detach().cpu().clone(),
                "update_count": int(entry["update_count"]),
                "point_count_total": int(entry["point_count_total"]),
                "last_batch_idx": int(entry["last_batch_idx"]),
            }

        return {
            "feature_dim": self.feature_dim,
            "momentum": self.momentum,
            "ordered": self.ordered,
            "device": str(self.device),
            "bank": bank_state,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """从状态字典恢复 bank。"""
        self.feature_dim = int(state["feature_dim"])
        self.momentum = float(state["momentum"])
        saved_ordered = bool(state.get("ordered", True))
        if not saved_ordered:
            # 当前仓库后面的第三、四步都依赖 ordered prototype；
            # 因此这里明确拒绝旧的 unordered 状态，免得把语义错误带到后面。
            raise ValueError("Only ordered boundary prototype banks are supported now.")
        self.ordered = True
        if "device" in state:
            self.device = torch.device(state["device"])

        loaded_bank = state["bank"]
        if not isinstance(loaded_bank, dict):
            raise TypeError("state['bank'] must be a dictionary")

        self.bank = {}
        for raw_key, entry in loaded_bank.items():
            boundary_key = self._normalize_key(raw_key[0], raw_key[1])
            prototype = entry["prototype"]
            if not isinstance(prototype, torch.Tensor):
                raise TypeError(f"loaded prototype for boundary {boundary_key} must be a torch.Tensor")
            prototype = prototype.to(self.device, dtype=torch.float32)
            prototype = _l2_normalize_vector(prototype)
            self.bank[boundary_key] = {
                "prototype": prototype,
                "update_count": int(entry["update_count"]),
                "point_count_total": int(entry["point_count_total"]),
                "last_batch_idx": int(entry["last_batch_idx"]),
            }

    def summary(self) -> BankSummaryDict:
        """返回当前 bank 的摘要统计信息。"""
        summary_dict: BankSummaryDict = {}
        for boundary_key in sorted(self.bank.keys()):
            entry = self.bank[boundary_key]
            prototype = entry["prototype"]
            if not isinstance(prototype, torch.Tensor):
                raise TypeError(f"stored prototype for boundary {boundary_key} must be a torch.Tensor")
            summary_dict[boundary_key] = {
                "prototype_shape": tuple(prototype.shape),
                "prototype_norm": float(prototype.norm(p=2).item()),
                "update_count": int(entry["update_count"]),
                "point_count_total": int(entry["point_count_total"]),
                "last_batch_idx": int(entry["last_batch_idx"]),
            }
        return summary_dict


def save_boundary_prototype_bank(bank: DynamicBoundaryPrototypeBank, save_path: str | Path) -> None:
    """保存动态 boundary prototype bank。

    输入：
        bank: DynamicBoundaryPrototypeBank
            当前的动态 bank 对象。
        save_path: str | Path
            保存路径。
    """
    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank.state_dict(), save_path)


def load_boundary_prototype_bank(
    load_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> DynamicBoundaryPrototypeBank:
    """加载动态 boundary prototype bank。

    输入：
        load_path: str | Path
            保存文件路径。
        map_location: str | torch.device
            加载时映射到的设备。

    输出：
        bank: DynamicBoundaryPrototypeBank
            恢复好的 bank 对象。
    """
    state = torch.load(Path(load_path).expanduser().resolve(), map_location=map_location)
    bank = DynamicBoundaryPrototypeBank(
        feature_dim=int(state["feature_dim"]),
        momentum=float(state["momentum"]),
        device=map_location,
    )
    bank.load_state_dict(state)
    return bank


def inspect_boundary_prototype_bank(bank: DynamicBoundaryPrototypeBank) -> None:
    """打印当前 bank 中各边界类别的统计信息。"""
    summary_dict = bank.summary()
    print("=" * 80)
    print("Dynamic Boundary Prototype Bank Summary")
    print("=" * 80)
    if len(summary_dict) == 0:
        print("Bank is empty.")
        return

    for boundary_key in sorted(summary_dict.keys()):
        stats = summary_dict[boundary_key]
        print(
            f"boundary={boundary_key!s:>8s} | "
            f"shape={stats['prototype_shape']!s:>8s} | "
            f"norm={float(stats['prototype_norm']):.6f} | "
            f"updates={int(stats['update_count']):4d} | "
            f"points={int(stats['point_count_total']):4d} | "
            f"last_batch={int(stats['last_batch_idx']):3d}"
        )


def compare_boundary_prototype_banks(
    bank_a: DynamicBoundaryPrototypeBank,
    bank_b: DynamicBoundaryPrototypeBank,
    atol: float = 1e-6,
) -> bool:
    """比较两个 bank 是否在数值上等价。

    输入：
        bank_a: DynamicBoundaryPrototypeBank
            第一个 bank。
        bank_b: DynamicBoundaryPrototypeBank
            第二个 bank。
        atol: float
            prototype 比较时的绝对误差容忍度。

    输出：
        is_equal: bool
            若两个 bank 的配置与各边界类别状态都一致，则返回 True。
    """
    if bank_a.feature_dim != bank_b.feature_dim:
        return False
    if bank_a.ordered != bank_b.ordered:
        return False
    if abs(bank_a.momentum - bank_b.momentum) > atol:
        return False
    if set(bank_a.bank.keys()) != set(bank_b.bank.keys()):
        return False

    for boundary_key in bank_a.bank.keys():
        entry_a = bank_a.bank[boundary_key]
        entry_b = bank_b.bank[boundary_key]
        proto_a = entry_a["prototype"]
        proto_b = entry_b["prototype"]
        if not isinstance(proto_a, torch.Tensor) or not isinstance(proto_b, torch.Tensor):
            return False
        if not torch.allclose(proto_a.cpu(), proto_b.cpu(), atol=atol, rtol=0.0):
            return False
        if int(entry_a["update_count"]) != int(entry_b["update_count"]):
            return False
        if int(entry_a["point_count_total"]) != int(entry_b["point_count_total"]):
            return False
        if int(entry_a["last_batch_idx"]) != int(entry_b["last_batch_idx"]):
            return False

    return True


def update_boundary_prototype_bank_from_filtered_points(
    bank: DynamicBoundaryPrototypeBank,
    filtered_boundary_dict: Dict[BoundaryKey, Dict[str, torch.Tensor | int | float | bool]],
    feature_already_normalized: bool = True,
) -> Tuple[DynamicBoundaryPrototypeBank, Dict[str, object]]:
    """从第一部分输出的高质量边界点更新动态 bank。

    输入：
        bank: DynamicBoundaryPrototypeBank
            当前动态 bank。
        filtered_boundary_dict: dict
            第一部分输出的高质量边界点字典。
        feature_already_normalized: bool
            点特征是否已经做过 L2 normalize。

    输出：
        bank: DynamicBoundaryPrototypeBank
            更新后的 bank。
        update_stats: dict
            当前这次更新的中间统计信息。
    """
    image_proto_dict = compute_image_level_boundary_prototypes(
        filtered_boundary_dict=filtered_boundary_dict,
        feature_already_normalized=feature_already_normalized,
    )
    bank_update_info = bank.update_from_image_prototypes(image_proto_dict)
    update_stats = {
        "num_registered_boundaries": len(bank.bank),
        "num_image_level_updates": sum(len(entries) for entries in image_proto_dict.values()),
        "image_proto_dict": image_proto_dict,
        "bank_update_info": bank_update_info,
    }
    return bank, update_stats


def _make_normalized_feature_block(
    num_points: int,
    feature_dim: int,
    center_seed: int,
    noise_scale: float = 0.08,
) -> torch.Tensor:
    """构造一组已经做过 L2 normalize 的 dummy 点特征。"""
    generator = torch.Generator().manual_seed(center_seed)
    center = torch.randn(feature_dim, generator=generator)
    center = _l2_normalize_vector(center)
    features = center.unsqueeze(0).repeat(num_points, 1)
    noise = torch.randn(num_points, feature_dim, generator=generator) * noise_scale
    features = features + noise
    features = F.normalize(features, p=2, dim=1, eps=1e-12)
    return features


def _make_coords_for_batch(batch_idx: int, num_points: int, y_start: int, x_start: int) -> torch.Tensor:
    """构造一组 dummy 坐标 [batch_idx, y, x]。"""
    ys = torch.arange(y_start, y_start + num_points, dtype=torch.long)
    xs = torch.arange(x_start, x_start + num_points, dtype=torch.long)
    batch_column = torch.full((num_points,), int(batch_idx), dtype=torch.long)
    coords = torch.stack([batch_column, ys, xs], dim=1)
    return coords


def build_dummy_filtered_boundary_dict_round1(feature_dim: int) -> Dict[BoundaryKey, Dict[str, torch.Tensor | int]]:
    """构造第一轮 dummy filtered_boundary_dict。"""
    boundary_dict: Dict[BoundaryKey, Dict[str, torch.Tensor | int]] = {}

    coords_12 = torch.cat(
        [
            _make_coords_for_batch(batch_idx=0, num_points=4, y_start=10, x_start=20),
            _make_coords_for_batch(batch_idx=1, num_points=5, y_start=30, x_start=40),
        ],
        dim=0,
    )
    feats_12 = torch.cat(
        [
            _make_normalized_feature_block(num_points=4, feature_dim=feature_dim, center_seed=12),
            _make_normalized_feature_block(num_points=5, feature_dim=feature_dim, center_seed=12),
        ],
        dim=0,
    )
    boundary_dict[(1, 2)] = {
        "coords": coords_12,
        "features": feats_12,
        "raw_count": int(coords_12.size(0)),
        "kept_count": int(coords_12.size(0)),
    }

    coords_10 = torch.cat(
        [
            _make_coords_for_batch(batch_idx=0, num_points=3, y_start=50, x_start=15),
            _make_coords_for_batch(batch_idx=1, num_points=4, y_start=60, x_start=18),
        ],
        dim=0,
    )
    feats_10 = torch.cat(
        [
            _make_normalized_feature_block(num_points=3, feature_dim=feature_dim, center_seed=10),
            _make_normalized_feature_block(num_points=4, feature_dim=feature_dim, center_seed=10),
        ],
        dim=0,
    )
    boundary_dict[(1, 0)] = {
        "coords": coords_10,
        "features": feats_10,
        "raw_count": int(coords_10.size(0)),
        "kept_count": int(coords_10.size(0)),
    }

    coords_23 = _make_coords_for_batch(batch_idx=1, num_points=6, y_start=70, x_start=35)
    feats_23 = _make_normalized_feature_block(num_points=6, feature_dim=feature_dim, center_seed=23)
    boundary_dict[(2, 3)] = {
        "coords": coords_23,
        "features": feats_23,
        "raw_count": int(coords_23.size(0)),
        "kept_count": int(coords_23.size(0)),
    }

    return boundary_dict


def build_dummy_filtered_boundary_dict_round2(feature_dim: int) -> Dict[BoundaryKey, Dict[str, torch.Tensor | int]]:
    """构造第二轮 dummy filtered_boundary_dict。"""
    boundary_dict: Dict[BoundaryKey, Dict[str, torch.Tensor | int]] = {}

    coords_21 = torch.cat(
        [
            _make_coords_for_batch(batch_idx=0, num_points=5, y_start=14, x_start=24),
            _make_coords_for_batch(batch_idx=2, num_points=4, y_start=34, x_start=44),
        ],
        dim=0,
    )
    feats_21 = torch.cat(
        [
            _make_normalized_feature_block(num_points=5, feature_dim=feature_dim, center_seed=12, noise_scale=0.10),
            _make_normalized_feature_block(num_points=4, feature_dim=feature_dim, center_seed=12, noise_scale=0.10),
        ],
        dim=0,
    )
    boundary_dict[(2, 1)] = {
        "coords": coords_21,
        "features": feats_21,
        "raw_count": int(coords_21.size(0)),
        "kept_count": int(coords_21.size(0)),
    }

    coords_10 = _make_coords_for_batch(batch_idx=2, num_points=5, y_start=80, x_start=22)
    feats_10 = _make_normalized_feature_block(num_points=5, feature_dim=feature_dim, center_seed=10, noise_scale=0.10)
    boundary_dict[(1, 0)] = {
        "coords": coords_10,
        "features": feats_10,
        "raw_count": int(coords_10.size(0)),
        "kept_count": int(coords_10.size(0)),
    }

    coords_30 = _make_coords_for_batch(batch_idx=0, num_points=4, y_start=90, x_start=28)
    feats_30 = _make_normalized_feature_block(num_points=4, feature_dim=feature_dim, center_seed=30)
    boundary_dict[(3, 0)] = {
        "coords": coords_30,
        "features": feats_30,
        "raw_count": int(coords_30.size(0)),
        "kept_count": int(coords_30.size(0)),
    }

    return boundary_dict


def main() -> None:
    """最小可运行 demo。"""
    feature_dim = 32
    save_path = Path(__file__).resolve().parent / "outputs" / "dynamic_boundary_prototype_bank_demo.pth"

    bank = DynamicBoundaryPrototypeBank(
        feature_dim=feature_dim,
        momentum=0.9,
        device="cpu",
    )

    print("\n[Round 1] Update bank with first dummy batch")
    round1_boundary_dict = build_dummy_filtered_boundary_dict_round1(feature_dim=feature_dim)
    bank, round1_stats = update_boundary_prototype_bank_from_filtered_points(
        bank=bank,
        filtered_boundary_dict=round1_boundary_dict,
        feature_already_normalized=True,
    )
    print(f"Round 1 registered boundaries: {round1_stats['num_registered_boundaries']}")
    print(f"Round 1 image-level updates: {round1_stats['num_image_level_updates']}")
    inspect_boundary_prototype_bank(bank)

    print("\n[Round 2] Update bank with second dummy batch")
    round2_boundary_dict = build_dummy_filtered_boundary_dict_round2(feature_dim=feature_dim)
    bank, round2_stats = update_boundary_prototype_bank_from_filtered_points(
        bank=bank,
        filtered_boundary_dict=round2_boundary_dict,
        feature_already_normalized=True,
    )
    print(f"Round 2 registered boundaries: {round2_stats['num_registered_boundaries']}")
    print(f"Round 2 image-level updates: {round2_stats['num_image_level_updates']}")
    inspect_boundary_prototype_bank(bank)

    save_boundary_prototype_bank(bank, save_path=save_path)
    print(f"\nBank saved to: {save_path}")

    loaded_bank = load_boundary_prototype_bank(load_path=save_path, map_location="cpu")
    print("\n[Loaded Bank] Summary after reload")
    inspect_boundary_prototype_bank(loaded_bank)

    is_consistent = compare_boundary_prototype_banks(bank, loaded_bank)
    print("\nState consistency check:", is_consistent)


if __name__ == "__main__":
    main()
