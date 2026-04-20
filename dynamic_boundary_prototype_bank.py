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
    if vector.dim() != 1:  # 这里只处理单个 prototype 向量，别混成矩阵。
        raise ValueError("vector must have shape (C,)")
    return F.normalize(vector.unsqueeze(0), p=2, dim=1, eps=eps).squeeze(0)  # 直接复用 PyTorch 的归一化实现。


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
                    "coords": Tensor[N, 3],      # [batch_idx, y, x]
                    "features": Tensor[N, C],    # 当前边界类别的点特征
                    ...
                }
            }
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
    image_proto_dict: ImagePrototypeDict = {}  # 最终的图级 prototype 结果统一放在这里。

    for raw_key, boundary_data in filtered_boundary_dict.items():  # 每个有序边界类别单独处理，(A,B) 与 (B,A) 绝不混合。
        boundary_key = canonicalize_ordered_boundary_key(raw_key[0], raw_key[1])  # 先把 key 规范化成有序表示。
        coords = boundary_data.get("coords")  # 当前边界类别的坐标张量。
        features = boundary_data.get("features")  # 当前边界类别的点特征张量。

        if not isinstance(coords, torch.Tensor):  # coords 缺失或类型不对时直接报错，别让后面静悄悄地坏掉。
            raise TypeError(f"filtered_boundary_dict[{boundary_key}]['coords'] must be a torch.Tensor")
        if not isinstance(features, torch.Tensor):  # 当前阶段优先支持直接用 features，避免重复提取。
            raise TypeError(f"filtered_boundary_dict[{boundary_key}]['features'] must be a torch.Tensor")
        if coords.dim() != 2 or coords.size(1) != 3:  # 坐标必须是 [N, 3]。
            raise ValueError(f"coords for boundary {boundary_key} must have shape (N, 3)")
        if features.dim() != 2:  # 特征必须是 [N, C]。
            raise ValueError(f"features for boundary {boundary_key} must have shape (N, C)")
        if coords.size(0) != features.size(0):  # 点数必须一一对应，不然就没法分组。
            raise ValueError(f"coords and features for boundary {boundary_key} must share the same first dimension")
        if coords.numel() == 0:  # 空类别直接跳过，没必要硬算。
            continue

        batch_indices = coords[:, 0].long()  # 第一列就是 batch_idx。
        unique_batch_indices = torch.unique(batch_indices)  # 当前边界类别在哪些图像里出现了，先找出来。
        image_proto_entries: List[Dict[str, torch.Tensor | int]] = []  # 当前边界类别对应的一组图级 prototype 条目。

        for batch_idx in unique_batch_indices.tolist():  # 按图像逐个处理，这正是当前函数要做的事。
            image_mask = batch_indices == int(batch_idx)  # 当前图像对应的点先筛出来。
            image_features = features[image_mask]  # 当前图像、当前边界类别的全部点特征。
            if image_features.size(0) == 0:  # 理论上不会进来，但还是兜一下。
                continue

            if not feature_already_normalized:  # 只有在点特征没归一化时，才在这里补一次 L2 normalize。
                image_features = F.normalize(image_features, p=2, dim=1, eps=1e-12)

            image_proto = image_features.mean(dim=0)  # 图内 prototype 先直接做均值，简单直接。
            image_proto = _l2_normalize_vector(image_proto)  # 图级 prototype 在求均值后必须再做一次归一化。

            image_proto_entries.append(
                {
                    "batch_idx": int(batch_idx),
                    "image_proto": image_proto,
                    "num_points": int(image_features.size(0)),
                }
            )  # 当前图像对应的 prototype 条目写进去，后面给 bank 做 EMA 更新。

        if len(image_proto_entries) > 0:  # 当前边界类别确实在这个 batch 里出现过时才写入结果。
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
        if feature_dim < 1:  # 特征维度至少得是正数，不然就没意义了。
            raise ValueError("feature_dim must be >= 1")
        if not (0.0 <= momentum < 1.0):  # momentum 取 [0,1) 就够了，1.0 会导致新信息完全进不来。
            raise ValueError("momentum must be in the range [0, 1)")

        self.feature_dim = int(feature_dim)  # 把特征维度记下来，后面做一致性检查要用。
        self.momentum = float(momentum)  # EMA 动量系数也存下来。
        self.ordered = True  # 当前项目只允许 ordered boundary，这里固定为 True 只是为了把状态写清楚。
        self.device = torch.device(device) if device is not None else torch.device("cpu")  # 默认放 CPU，最稳。
        self.bank: Dict[BoundaryKey, Dict[str, torch.Tensor | int]] = {}  # bank 本体就是动态字典，这里不预先枚举任何边界类别。

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
        update_info: Dict[BoundaryKey, Dict[str, int | float]] = {}  # 当前这次更新的摘要信息统一放这里。

        for raw_key, image_entries in image_proto_dict.items():  # 每个边界类别单独处理，互不干扰。
            boundary_key = self._normalize_key(raw_key[0], raw_key[1])  # 先统一 key，别让同一类边界分叉。
            for entry in image_entries:  # 同一边界类别可能在一个 batch 的多张图里都出现，所以逐个图级 prototype 更新。
                image_proto = entry["image_proto"]  # 当前图像的 boundary prototype。
                batch_idx = entry["batch_idx"]  # 当前图像所在 batch 索引。
                num_points = entry["num_points"]  # 当前图像当前边界类别参与均值的点数。

                if not isinstance(image_proto, torch.Tensor):  # prototype 必须是 tensor，不然没法做 EMA。
                    raise TypeError(f"image_proto for boundary {boundary_key} must be a torch.Tensor")
                if image_proto.dim() != 1 or image_proto.numel() != self.feature_dim:  # prototype 维度必须和 bank 预设一致。
                    raise ValueError(
                        f"image_proto for boundary {boundary_key} must have shape ({self.feature_dim},)"
                    )

                image_proto = image_proto.to(self.device, dtype=torch.float32)  # 当前图级 prototype 搬到 bank 设备上，统一类型。
                image_proto = _l2_normalize_vector(image_proto)  # 图级 prototype 理论上已经归一化过，但这里再稳一下不吃亏。

                if boundary_key not in self.bank:  # bank 中没有该 key 时，直接动态注册。
                    bank_prototype = image_proto.clone()  # 首次出现时直接把图级 prototype 作为 bank prototype。
                    update_count = 1  # 这个类别第一次被更新。
                    point_count_total = int(num_points)  # 累计参与点数就从当前图像开始。
                else:  # bank 中已有该 key 时，按 EMA 规则更新。
                    old_prototype = self.bank[boundary_key]["prototype"]  # 旧 prototype 先拿出来。
                    if not isinstance(old_prototype, torch.Tensor):  # 理论兜底，防止 bank 状态被错误污染。
                        raise TypeError(f"stored prototype for boundary {boundary_key} must be a torch.Tensor")
                    bank_prototype = self.momentum * old_prototype + (1.0 - self.momentum) * image_proto  # 标准 EMA 更新。
                    bank_prototype = _l2_normalize_vector(bank_prototype)  # EMA 后再做一次归一化，保证后续余弦匹配稳定。
                    update_count = int(self.bank[boundary_key]["update_count"]) + 1  # 更新次数加一。
                    point_count_total = int(self.bank[boundary_key]["point_count_total"]) + int(num_points)  # 累计点数继续累加。

                self.bank[boundary_key] = {
                    "prototype": bank_prototype,
                    "update_count": update_count,
                    "point_count_total": point_count_total,
                    "last_batch_idx": int(batch_idx),
                }  # 当前边界类别在 bank 中的最新状态回写进去。

                update_info[boundary_key] = {
                    "update_count": update_count,
                    "point_count_total": point_count_total,
                    "prototype_norm": float(bank_prototype.norm(p=2).item()),
                }  # 当前这次更新后的摘要信息顺手记录下来，外部日志会方便很多。

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
        boundary_key = self._normalize_key(a, b)  # 访问前统一做 key 规范化。
        if boundary_key not in self.bank:  # 不存在就返回 None，调用方自己决定怎么处理。
            return None
        prototype = self.bank[boundary_key]["prototype"]  # 取出 prototype。
        if not isinstance(prototype, torch.Tensor):  # 理论兜底。
            raise TypeError(f"stored prototype for boundary {boundary_key} must be a torch.Tensor")
        return prototype

    def state_dict(self) -> Dict[str, object]:
        """返回可保存的 bank 状态字典。"""
        bank_state: Dict[BoundaryKey, Dict[str, torch.Tensor | int]] = {}  # 单独构造可序列化状态。
        for boundary_key, entry in self.bank.items():  # 每个边界类别单独整理，逻辑最透明。
            prototype = entry["prototype"]  # 当前类别的 prototype。
            if not isinstance(prototype, torch.Tensor):  # 理论兜底，避免保存脏状态。
                raise TypeError(f"stored prototype for boundary {boundary_key} must be a torch.Tensor")
            bank_state[boundary_key] = {
                "prototype": prototype.detach().cpu().clone(),  # 保存时统一拷回 CPU，最稳妥。
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
        }  # 元信息和 bank 本体都一起存进去，后面恢复才完整。

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """从状态字典恢复 bank。"""
        self.feature_dim = int(state["feature_dim"])  # 先恢复特征维度。
        self.momentum = float(state["momentum"])  # 再恢复 EMA 动量。
        saved_ordered = bool(state.get("ordered", True))  # 旧状态里如果带了 ordered 字段，这里也读出来。
        if not saved_ordered:  # 当前代码已经只支持 ordered boundary，旧的 unordered 状态不允许继续混用。
            raise ValueError("Only ordered boundary prototype banks are supported now.")
        self.ordered = True  # 恢复后仍然固定成 True，避免任何无序逻辑回流。
        if "device" in state:  # 如果状态里带了 device 字段，就把它也恢复出来。
            self.device = torch.device(state["device"])

        loaded_bank = state["bank"]  # 取出真正的 bank 状态。
        if not isinstance(loaded_bank, dict):  # 状态格式不对时直接报错。
            raise TypeError("state['bank'] must be a dictionary")

        self.bank = {}  # 先清空当前 bank，避免旧状态污染。
        for raw_key, entry in loaded_bank.items():  # 每个边界类别逐个恢复。
            boundary_key = self._normalize_key(raw_key[0], raw_key[1])  # 恢复时仍然走同样的 key 规范化逻辑。
            prototype = entry["prototype"]  # 当前类别 prototype。
            if not isinstance(prototype, torch.Tensor):  # prototype 必须是 tensor。
                raise TypeError(f"loaded prototype for boundary {boundary_key} must be a torch.Tensor")
            prototype = prototype.to(self.device, dtype=torch.float32)  # 恢复到 bank 当前 device 上。
            prototype = _l2_normalize_vector(prototype)  # 再稳一下，确保 prototype 范数是 1。
            self.bank[boundary_key] = {
                "prototype": prototype,
                "update_count": int(entry["update_count"]),
                "point_count_total": int(entry["point_count_total"]),
                "last_batch_idx": int(entry["last_batch_idx"]),
            }  # 恢复当前边界类别的全部状态。

    def summary(self) -> BankSummaryDict:
        """返回当前 bank 的摘要统计信息。"""
        summary_dict: BankSummaryDict = {}  # 摘要信息统一放在这里。
        for boundary_key in sorted(self.bank.keys()):  # 按 key 排序输出，阅读会稳定很多。
            entry = self.bank[boundary_key]  # 当前边界类别的 bank 条目。
            prototype = entry["prototype"]  # 当前类别 prototype。
            if not isinstance(prototype, torch.Tensor):  # 理论兜底。
                raise TypeError(f"stored prototype for boundary {boundary_key} must be a torch.Tensor")
            summary_dict[boundary_key] = {
                "prototype_shape": tuple(prototype.shape),
                "prototype_norm": float(prototype.norm(p=2).item()),
                "update_count": int(entry["update_count"]),
                "point_count_total": int(entry["point_count_total"]),
                "last_batch_idx": int(entry["last_batch_idx"]),
            }  # 当前类别的关键统计项都写进去，外部检查会方便很多。
        return summary_dict


def save_boundary_prototype_bank(bank: DynamicBoundaryPrototypeBank, save_path: str | Path) -> None:
    """保存动态 boundary prototype bank。

    输入：
        bank: DynamicBoundaryPrototypeBank
            当前的动态 bank 对象。
        save_path: str | Path
            保存路径。
    """
    save_path = Path(save_path).expanduser().resolve()  # 路径先标准化。
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 输出目录不存在就创建。
    torch.save(bank.state_dict(), save_path)  # 直接保存整个 state_dict，简单而且稳。


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
    state = torch.load(Path(load_path).expanduser().resolve(), map_location=map_location)  # 先把保存状态读回来。
    bank = DynamicBoundaryPrototypeBank(
        feature_dim=int(state["feature_dim"]),
        momentum=float(state["momentum"]),
        device=map_location,
    )  # 先按元信息构造一个同配置的空 bank。
    bank.load_state_dict(state)  # 再把状态灌进去。
    return bank


def inspect_boundary_prototype_bank(bank: DynamicBoundaryPrototypeBank) -> None:
    """打印当前 bank 中各边界类别的统计信息。"""
    summary_dict = bank.summary()  # 先拿到摘要信息，打印逻辑会更清楚。
    print("=" * 80)  # 分隔线先打上。
    print("Dynamic Boundary Prototype Bank Summary")  # 标题直接写明白。
    print("=" * 80)
    if len(summary_dict) == 0:  # 空 bank 时直接说明情况。
        print("Bank is empty.")
        return

    for boundary_key in sorted(summary_dict.keys()):  # 按 key 排序，输出顺序稳定。
        stats = summary_dict[boundary_key]  # 当前边界类别的统计项。
        print(
            f"boundary={boundary_key!s:>8s} | "
            f"shape={stats['prototype_shape']!s:>8s} | "
            f"norm={float(stats['prototype_norm']):.6f} | "
            f"updates={int(stats['update_count']):4d} | "
            f"points={int(stats['point_count_total']):4d} | "
            f"last_batch={int(stats['last_batch_idx']):3d}"
        )  # 你关心的 key、范数、更新次数、累计点数都放齐。


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
    if bank_a.feature_dim != bank_b.feature_dim:  # 特征维度不一样时直接判不相等。
        return False
    if bank_a.ordered != bank_b.ordered:  # ordered 设置不同也直接判不相等。
        return False
    if abs(bank_a.momentum - bank_b.momentum) > atol:  # EMA 动量差太大说明配置不一致。
        return False
    if set(bank_a.bank.keys()) != set(bank_b.bank.keys()):  # 注册的边界类别集合不同，也不用比了。
        return False

    for boundary_key in bank_a.bank.keys():  # 每个边界类别逐个检查。
        entry_a = bank_a.bank[boundary_key]  # bank_a 当前类别状态。
        entry_b = bank_b.bank[boundary_key]  # bank_b 当前类别状态。
        proto_a = entry_a["prototype"]  # bank_a prototype。
        proto_b = entry_b["prototype"]  # bank_b prototype。
        if not isinstance(proto_a, torch.Tensor) or not isinstance(proto_b, torch.Tensor):  # 理论兜底。
            return False
        if not torch.allclose(proto_a.cpu(), proto_b.cpu(), atol=atol, rtol=0.0):  # prototype 用 allclose 比较才合理。
            return False
        if int(entry_a["update_count"]) != int(entry_b["update_count"]):  # 更新次数必须一致。
            return False
        if int(entry_a["point_count_total"]) != int(entry_b["point_count_total"]):  # 累计点数也必须一致。
            return False
        if int(entry_a["last_batch_idx"]) != int(entry_b["last_batch_idx"]):  # 最后更新的 batch_idx 也应该一致。
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
    )  # 先把高质量点按“图像”为单位汇总成图级 prototype。
    bank_update_info = bank.update_from_image_prototypes(image_proto_dict)  # 再把这些图级 prototype 送进动态 bank 做 EMA 更新。
    update_stats = {
        "num_registered_boundaries": len(bank.bank),
        "num_image_level_updates": sum(len(entries) for entries in image_proto_dict.values()),
        "image_proto_dict": image_proto_dict,
        "bank_update_info": bank_update_info,
    }  # 把这次更新的关键统计一起返回，后面记录日志或调试都会方便。
    return bank, update_stats


def _make_normalized_feature_block(
    num_points: int,
    feature_dim: int,
    center_seed: int,
    noise_scale: float = 0.08,
) -> torch.Tensor:
    """构造一组已经做过 L2 normalize 的 dummy 点特征。"""
    generator = torch.Generator().manual_seed(center_seed)  # 每个类别块单独固定随机种子，结果更稳定。
    center = torch.randn(feature_dim, generator=generator)  # 先随机生成一个中心向量。
    center = _l2_normalize_vector(center)  # 中心也先归一化，后面加噪声会更自然。
    features = center.unsqueeze(0).repeat(num_points, 1)  # 所有点先围绕同一个中心展开。
    noise = torch.randn(num_points, feature_dim, generator=generator) * noise_scale  # 加一点小噪声，让点云别显得太死板。
    features = features + noise  # 把噪声叠上去。
    features = F.normalize(features, p=2, dim=1, eps=1e-12)  # 这里假设输出点特征已经归一化过，和题目设定对齐。
    return features


def _make_coords_for_batch(batch_idx: int, num_points: int, y_start: int, x_start: int) -> torch.Tensor:
    """构造一组 dummy 坐标 [batch_idx, y, x]。"""
    ys = torch.arange(y_start, y_start + num_points, dtype=torch.long)  # y 坐标简单按连续整数生成。
    xs = torch.arange(x_start, x_start + num_points, dtype=torch.long)  # x 坐标也做一个连续偏移。
    batch_column = torch.full((num_points,), int(batch_idx), dtype=torch.long)  # batch_idx 单独拉一列。
    coords = torch.stack([batch_column, ys, xs], dim=1)  # 最后堆成 [N, 3] 形式。
    return coords


def build_dummy_filtered_boundary_dict_round1(feature_dim: int) -> Dict[BoundaryKey, Dict[str, torch.Tensor | int]]:
    """构造第一轮 dummy filtered_boundary_dict。"""
    boundary_dict: Dict[BoundaryKey, Dict[str, torch.Tensor | int]] = {}  # 第一轮的 dummy 输入统一放这里。

    coords_12 = torch.cat(
        [
            _make_coords_for_batch(batch_idx=0, num_points=4, y_start=10, x_start=20),
            _make_coords_for_batch(batch_idx=1, num_points=5, y_start=30, x_start=40),
        ],
        dim=0,
    )  # (1,2) 这个边界类别让它同时出现在 batch 0 和 batch 1。
    feats_12 = torch.cat(
        [
            _make_normalized_feature_block(num_points=4, feature_dim=feature_dim, center_seed=12),
            _make_normalized_feature_block(num_points=5, feature_dim=feature_dim, center_seed=12),
        ],
        dim=0,
    )  # 同一边界类别的不同图像共享一个大致中心，只加一点轻噪声。
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
    )  # (1,0) 同样做成跨图像出现。
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

    coords_23 = _make_coords_for_batch(batch_idx=1, num_points=6, y_start=70, x_start=35)  # (2,3) 只放在 batch 1 里出现一次。
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
    boundary_dict: Dict[BoundaryKey, Dict[str, torch.Tensor | int]] = {}  # 第二轮 dummy 输入。

    coords_21 = torch.cat(
        [
            _make_coords_for_batch(batch_idx=0, num_points=5, y_start=14, x_start=24),
            _make_coords_for_batch(batch_idx=2, num_points=4, y_start=34, x_start=44),
        ],
        dim=0,
    )  # 故意写成 (2,1)，用来验证 ordered 版本下它会作为独立边界类别保留下来。
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

    coords_10 = _make_coords_for_batch(batch_idx=2, num_points=5, y_start=80, x_start=22)  # (1,0) 再来一次，用于测试 EMA 累积更新。
    feats_10 = _make_normalized_feature_block(num_points=5, feature_dim=feature_dim, center_seed=10, noise_scale=0.10)
    boundary_dict[(1, 0)] = {
        "coords": coords_10,
        "features": feats_10,
        "raw_count": int(coords_10.size(0)),
        "kept_count": int(coords_10.size(0)),
    }

    coords_30 = _make_coords_for_batch(batch_idx=0, num_points=4, y_start=90, x_start=28)  # (3,0) 是一个新类别，用来测试动态注册。
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
    feature_dim = 32  # dummy 特征维度设成 32，和你的设定一致。
    save_path = Path(__file__).resolve().parent / "outputs" / "dynamic_boundary_prototype_bank_demo.pth"  # demo 保存路径放到项目 outputs 目录。

    bank = DynamicBoundaryPrototypeBank(
        feature_dim=feature_dim,
        momentum=0.9,
        device="cpu",
    )  # 初始化一个动态 bank，固定按有序边界管理。

    print("\n[Round 1] Update bank with first dummy batch")  # 第一轮更新标题。
    round1_boundary_dict = build_dummy_filtered_boundary_dict_round1(feature_dim=feature_dim)  # 先构造第一批 dummy 边界点。
    bank, round1_stats = update_boundary_prototype_bank_from_filtered_points(
        bank=bank,
        filtered_boundary_dict=round1_boundary_dict,
        feature_already_normalized=True,
    )  # 第一轮更新，点特征假设已经归一化过。
    print(f"Round 1 registered boundaries: {round1_stats['num_registered_boundaries']}")  # 打印当前已注册边界类别数。
    print(f"Round 1 image-level updates: {round1_stats['num_image_level_updates']}")  # 打印这轮图级更新次数。
    inspect_boundary_prototype_bank(bank)  # 检查第一轮后的 bank 状态。

    print("\n[Round 2] Update bank with second dummy batch")  # 第二轮更新标题。
    round2_boundary_dict = build_dummy_filtered_boundary_dict_round2(feature_dim=feature_dim)  # 构造第二批 dummy 数据。
    bank, round2_stats = update_boundary_prototype_bank_from_filtered_points(
        bank=bank,
        filtered_boundary_dict=round2_boundary_dict,
        feature_already_normalized=True,
    )  # 第二轮更新，测试 EMA 和动态注册是否正常。
    print(f"Round 2 registered boundaries: {round2_stats['num_registered_boundaries']}")  # 当前总注册类别数。
    print(f"Round 2 image-level updates: {round2_stats['num_image_level_updates']}")  # 当前轮图级更新次数。
    inspect_boundary_prototype_bank(bank)  # 再检查第二轮后的 bank 状态。

    save_boundary_prototype_bank(bank, save_path=save_path)  # 把当前 bank 保存到磁盘。
    print(f"\nBank saved to: {save_path}")  # 打印保存路径，方便你后面直接看文件。

    loaded_bank = load_boundary_prototype_bank(load_path=save_path, map_location="cpu")  # 再从磁盘加载回来，测试恢复逻辑。
    print("\n[Loaded Bank] Summary after reload")  # 加载后检查标题。
    inspect_boundary_prototype_bank(loaded_bank)  # 打印加载后的摘要，看是否和保存前一致。

    is_consistent = compare_boundary_prototype_banks(bank, loaded_bank)  # 用真正合理的 tensor allclose 规则检查保存前后是否一致。
    print("\nState consistency check:", is_consistent)  # 最后把一致性检查结果打印出来。


if __name__ == "__main__":  # 让这个文件可以独立运行。
    main()
