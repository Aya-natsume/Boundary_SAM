"""Boundary_SAM 的简化训练脚本。

训练目标：
1. 源域 `t2w` 走学生源域编码器和共享解码器，直接和真实标签做交叉熵。
2. 目标域 `adc` 走教师目标域编码器和共享解码器，先得到粗伪标签。
3. 再用当前仓库第一到第四部分生成的 box / point prompts 驱动 SAM，得到目标域精修伪标签。
4. 最后让学生目标域编码器和共享解码器去拟合这个精修伪标签。

说明：
1. 这里按 `Boundary_v2_contrast_pic/train.py` 的整体结构，保留 `SE_source + SE_target + SD` 的分工。
2. 当前脚本只写最小可跑版本，不再额外叠加判别器、对比损失或复杂课程学习。
3. 这里默认依赖官方 `segment_anything` 包；如果环境里还没有，需要你自己先装好。
"""

from __future__ import annotations

import argparse
import copy
import itertools
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dynamic_boundary_prototype_bank import DynamicBoundaryPrototypeBank, compute_image_level_boundary_prototypes
from model.Seg import Decoder, Encoder, load_pretrained_weights
from ordered_boundary_point_prompt_generation import generate_point_prompts_for_box_list
from ordered_boundary_prompt_score import generate_ordered_core_points_for_boxes
from picai_dataset import PICAITrainDataset_with_transform, RandomGenerator
from source_fine_boundary_points import (
    BoundaryDict,
    assign_fine_boundary_labels,
    build_source_fine_boundary_points,
    canonicalize_ordered_boundary_key,
)


BoundaryKey = Tuple[int, int]
PromptResult = Dict[str, object]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Train Boundary_SAM with SAM-refined target pseudo labels.")
    parser.add_argument("--data-root", type=str, default="/home/chenxu/dataset/picai")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs" / "train"))
    parser.add_argument("--source-modality", type=str, default="t2w")
    parser.add_argument("--target-modality", type=str, default="adc")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--source-loss-weight", type=float, default=1.0)
    parser.add_argument("--target-loss-weight", type=float, default=0.1)
    parser.add_argument("--ema-alpha", type=float, default=0.99)
    parser.add_argument("--boundary-kernel", type=int, default=5)
    parser.add_argument("--neighbor-kernel", type=int, default=3)
    parser.add_argument("--keep-ratio", type=float, default=0.8)
    parser.add_argument("--min-boundary-points", type=int, default=10)
    parser.add_argument("--bank-momentum", type=float, default=0.95)
    parser.add_argument("--box-padding", type=int, default=6)
    parser.add_argument("--segments-per-strip", type=int, default=3)
    parser.add_argument("--prompt-min-score", type=float, default=None)
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--sam-checkpoint", type=str, required=True)
    parser.add_argument("--sam-device", type=str, default=None)
    parser.add_argument("--source-encoder-checkpoint", type=str, default=None)
    parser.add_argument("--target-encoder-checkpoint", type=str, default=None)
    parser.add_argument("--decoder-checkpoint", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机种子，减少训练过程里的无谓波动。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_ema_variables(student_model: nn.Module, teacher_model: nn.Module, alpha: float) -> None:
    """用 EMA 更新教师模型参数。"""
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1.0 - alpha)


def load_optional_checkpoint(
    module: nn.Module,
    checkpoint_path: Optional[str],
    map_location: str | torch.device = "cpu",
) -> None:
    """按需加载预训练权重。"""
    if checkpoint_path is None:
        return
    load_pretrained_weights(module, checkpoint_path, strict=True, map_location=map_location)


def _prototype_exists(prototype_bank: DynamicBoundaryPrototypeBank, a: int, b: int) -> bool:
    """检查某个 ordered prototype 是否已经存在。"""
    return prototype_bank.get(int(a), int(b)) is not None


def build_pair_boxes_from_boundary_dict(
    boundary_dict: BoundaryDict,
    height: int,
    width: int,
    padding: int,
    prototype_library: Optional[DynamicBoundaryPrototypeBank] = None,
    require_bidirectional: bool = True,
) -> List[Dict[str, object]]:
    """把 ordered boundary 点按 pair 聚合成粗 box。

    这里保留 pair 级粗定位的目的，不是为了直接在 box 里 top-k；
    而是给第三步一个局部几何范围，让后面的条带框构造仍然只看当前边界段。
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
                "boundary_coords": merged_coords,
                "ordered_keys": tuple(sorted(expected_ordered_keys)),
                "num_pixels": int(merged_coords.size(0)),
            }
        )
    return box_list


def build_box_core_list_from_strip_results(strip_result_list: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """把第三步结果整理成第四步可直接消费的输入。"""
    box_core_list: List[Dict[str, object]] = []
    for result in strip_result_list:
        pair = result.get("pair")
        box = result.get("box")
        core_ab = result.get("core_ab")
        core_ba = result.get("core_ba")
        if not isinstance(pair, tuple) or len(pair) != 2:
            continue
        if not isinstance(box, dict) or not isinstance(core_ab, dict) or not isinstance(core_ba, dict):
            continue
        # 第三步的调试字段需要继续透传给第四步和 SAM，可别在这里丢掉。
        prompt_box = dict(box)
        for key in (
            "boundary_points_global",
            "normal_samples_a",
            "normal_samples_b",
            "similarity_curve_a",
            "similarity_curve_b",
            "smooth_similarity_curve_a",
            "smooth_similarity_curve_b",
            "changepoint_index_a",
            "changepoint_index_b",
        ):
            if key in result and key not in prompt_box:
                prompt_box[key] = result[key]
        box_core_list.append(
            {
                "box": prompt_box,
                "core_ab": core_ab,
                "core_ba": core_ba,
                "a": int(pair[0]),
                "b": int(pair[1]),
            }
        )
    return box_core_list


def _prepare_sam_image(image_2d: torch.Tensor) -> np.ndarray:
    """把单通道医学切片转成 SAM 可消费的 3 通道 uint8 图像。"""
    image_np = image_2d.detach().cpu().to(torch.float32).numpy()
    image_min = float(image_np.min())
    image_max = float(image_np.max())
    if image_max <= image_min:
        image_np = np.zeros_like(image_np, dtype=np.uint8)
    else:
        image_np = (image_np - image_min) / (image_max - image_min + 1e-6)
        image_np = np.clip(image_np * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.repeat(image_np[..., None], 3, axis=2)


def _build_sam_point_inputs(prompt_result: PromptResult) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """把第四步结果整理成 SAM 的 point prompt 输入。"""
    points_a = prompt_result.get("points_a", [])
    points_b = prompt_result.get("points_b", [])
    if not isinstance(points_a, list) or not isinstance(points_b, list):
        return None, None

    point_coords: List[List[float]] = []
    point_labels: List[int] = []
    for point in points_a:
        if isinstance(point, dict):
            point_coords.append([float(point["x"]), float(point["y"])])
            point_labels.append(1)
    for point in points_b:
        if isinstance(point, dict):
            point_coords.append([float(point["x"]), float(point["y"])])
            point_labels.append(0)

    if len(point_coords) == 0:
        return None, None
    return np.asarray(point_coords, dtype=np.float32), np.asarray(point_labels, dtype=np.int64)


def _refine_single_sample_with_sam(
    predictor: object,
    image_2d: torch.Tensor,
    pseudo_label_2d: torch.Tensor,
    prompt_results: Sequence[PromptResult],
) -> torch.Tensor:
    """用当前 sample 的 prompts 驱动 SAM，得到局部精修后的伪标签。

    这里的更新策略刻意保持简单：
    1. SAM 只在当前 pair 的条带框附近发挥作用。
    2. 更新只落在 `strip_mask_box` 与当前 `{a, b}` 区域的交集里。
    3. SAM 二值 mask 为真时赋成 pair 的第一类 `a`，否则赋成第二类 `b`。
    """
    predictor.set_image(_prepare_sam_image(image_2d))
    refined_label = pseudo_label_2d.clone()

    for prompt_result in prompt_results:
        box = prompt_result["box"]
        x1, y1, x2, y2 = [int(v) for v in box["box"]]
        a = int(prompt_result["a"])
        b = int(prompt_result["b"])
        point_coords, point_labels = _build_sam_point_inputs(prompt_result)
        if point_coords is None or point_labels is None:
            continue

        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=np.asarray([x1, y1, x2, y2], dtype=np.float32),
            multimask_output=False,
        )
        sam_mask = torch.from_numpy(masks[0]).to(device=refined_label.device, dtype=torch.bool)
        local_mask = sam_mask[y1:y2 + 1, x1:x2 + 1]
        strip_mask = torch.as_tensor(
            prompt_result.get("strip_mask_box", box["strip_mask_box"]),
            device=refined_label.device,
            dtype=torch.bool,
        )

        local_label = refined_label[y1:y2 + 1, x1:x2 + 1]
        pair_region = strip_mask & ((local_label == int(a)) | (local_label == int(b)))
        if not bool(pair_region.any()):
            continue

        local_label[pair_region & local_mask] = int(a)
        local_label[pair_region & (~local_mask)] = int(b)
        refined_label[y1:y2 + 1, x1:x2 + 1] = local_label

    return refined_label


def refine_target_pseudo_labels_with_sam(
    predictor: object,
    target_images: torch.Tensor,
    target_pseudo_label: torch.Tensor,
    prompt_result_list: Sequence[PromptResult],
) -> torch.Tensor:
    """对一个 batch 的目标域粗伪标签做 SAM 精修。"""
    refined_labels: List[torch.Tensor] = []
    for batch_idx in range(int(target_images.size(0))):
        sample_prompt_results = [
            result for result in prompt_result_list
            if int(result["box"]["batch_idx"]) == int(batch_idx)
        ]
        refined_labels.append(
            _refine_single_sample_with_sam(
                predictor=predictor,
                image_2d=target_images[batch_idx, 0],
                pseudo_label_2d=target_pseudo_label[batch_idx],
                prompt_results=sample_prompt_results,
            )
        )
    return torch.stack(refined_labels, dim=0)


def build_sam_predictor(model_type: str, checkpoint: str, device: torch.device) -> object:
    """构建官方 SAM predictor。"""
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as error:
        raise ImportError(
            "segment_anything is not installed. Please install the official SAM package before running train.py."
        ) from error

    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device=device)
    sam_model.eval()
    return SamPredictor(sam_model)


def build_target_prompt_results(
    teacher_feature_map: torch.Tensor,
    teacher_prob_map: torch.Tensor,
    target_pseudo_label: torch.Tensor,
    prototype_bank: DynamicBoundaryPrototypeBank,
    boundary_kernel: int,
    neighbor_kernel: int,
    box_padding: int,
    segments_per_strip: int,
    prompt_min_score: Optional[float],
) -> List[PromptResult]:
    """从教师粗伪标签出发，完整跑通第三步和第四步。"""
    boundary_dict = assign_fine_boundary_labels(
        seg_label=target_pseudo_label,
        num_classes=int(teacher_prob_map.size(1)),
        boundary_kernel=boundary_kernel,
        neighbor_kernel=neighbor_kernel,
    )
    height = int(target_pseudo_label.size(1))
    width = int(target_pseudo_label.size(2))
    pair_box_list = build_pair_boxes_from_boundary_dict(
        boundary_dict=boundary_dict,
        height=height,
        width=width,
        padding=box_padding,
        prototype_library=prototype_bank,
        require_bidirectional=True,
    )
    if len(pair_box_list) == 0:
        return []

    strip_result_list = generate_ordered_core_points_for_boxes(
        feature_map=teacher_feature_map,
        box_list=pair_box_list,
        prototype_library=prototype_bank,
        prob_map=teacher_prob_map,
        use_soft_uncertainty=False,
    )
    if len(strip_result_list) == 0:
        return []

    box_core_list = build_box_core_list_from_strip_results(strip_result_list)
    if len(box_core_list) == 0:
        return []

    return generate_point_prompts_for_box_list(
        prob_map=teacher_prob_map,
        box_core_list=box_core_list,
        topk_per_side=segments_per_strip,
        min_score=prompt_min_score,
    )


def build_source_bank_from_batch(
    source_feature_map: torch.Tensor,
    source_label: torch.Tensor,
    prototype_bank: DynamicBoundaryPrototypeBank,
    num_classes: int,
    boundary_kernel: int,
    neighbor_kernel: int,
    keep_ratio: float,
    min_points: int,
) -> None:
    """用源域 GT 和学生源域特征更新 ordered prototype bank。"""
    with torch.no_grad():
        _, filtered_boundary_dict, _ = build_source_fine_boundary_points(
            feature_map=source_feature_map.detach(),
            seg_label=source_label,
            num_classes=num_classes,
            boundary_kernel=boundary_kernel,
            neighbor_kernel=neighbor_kernel,
            keep_ratio=keep_ratio,
            min_points=min_points,
        )
        image_proto_dict = compute_image_level_boundary_prototypes(
            filtered_boundary_dict=filtered_boundary_dict,
            feature_already_normalized=True,
        )
        if len(image_proto_dict) > 0:
            prototype_bank.update_from_image_prototypes(image_proto_dict)


def maybe_squeeze_label(label_tensor: torch.Tensor) -> torch.Tensor:
    """把 `[B, 1, H, W]` 标签兼容成 `[B, H, W]`。"""
    if label_tensor.dim() == 4 and label_tensor.size(1) == 1:
        return label_tensor[:, 0]
    return label_tensor


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    se_source: Encoder,
    se_target: Encoder,
    sd: Decoder,
    teacher_se_target: Encoder,
    teacher_sd: Decoder,
) -> None:
    """保存当前 epoch 的学生和教师权重。"""
    torch.save(se_source.state_dict(), output_dir / f"SE_source_{epoch}.pth")
    torch.save(se_target.state_dict(), output_dir / f"SE_target_{epoch}.pth")
    torch.save(sd.state_dict(), output_dir / f"SD_{epoch}.pth")
    torch.save(teacher_se_target.state_dict(), output_dir / f"SE_target_ema_{epoch}.pth")
    torch.save(teacher_sd.state_dict(), output_dir / f"SD_ema_{epoch}.pth")


def main() -> None:
    """训练入口。"""
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_device = torch.device(args.sam_device) if args.sam_device is not None else device

    source_transform = RandomGenerator(output_size=(192, 192), SpatialAug=True, IntensityAug=True, NonlinearAug=False)
    target_transform = RandomGenerator(output_size=(192, 192), SpatialAug=True, IntensityAug=True, NonlinearAug=False)

    source_train_loader = DataLoader(
        dataset=PICAITrainDataset_with_transform(
            data_root=Path(args.data_root) / "train",
            n_slices=1,
            modality=args.source_modality,
            transform=source_transform,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    target_train_loader = DataLoader(
        dataset=PICAITrainDataset_with_transform(
            data_root=Path(args.data_root) / "train",
            n_slices=1,
            modality=args.target_modality,
            transform=target_transform,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    se_source = Encoder().to(device)
    se_target = Encoder().to(device)
    sd = Decoder(num_class=args.num_classes).to(device)
    load_optional_checkpoint(se_source, args.source_encoder_checkpoint, map_location=device)
    load_optional_checkpoint(se_target, args.target_encoder_checkpoint, map_location=device)
    load_optional_checkpoint(sd, args.decoder_checkpoint, map_location=device)

    teacher_se_target = copy.deepcopy(se_target).to(device)
    teacher_sd = copy.deepcopy(sd).to(device)
    teacher_se_target.eval()
    teacher_sd.eval()
    for parameter in itertools.chain(teacher_se_target.parameters(), teacher_sd.parameters()):
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        itertools.chain(se_source.parameters(), se_target.parameters(), sd.parameters()),
        lr=args.lr,
    )
    criterion_ce = nn.CrossEntropyLoss().to(device)
    sam_predictor = build_sam_predictor(args.sam_model_type, args.sam_checkpoint, sam_device)
    prototype_bank = DynamicBoundaryPrototypeBank(feature_dim=32, momentum=args.bank_momentum, device=device)

    for epoch in range(args.epochs):
        se_source.train()
        se_target.train()
        sd.train()
        train_iterator = tqdm(
            zip(source_train_loader, target_train_loader),
            total=min(len(source_train_loader), len(target_train_loader)),
            ncols=180,
            desc=f"Epoch {epoch}",
        )

        for batch_source, batch_target in train_iterator:
            source_image = batch_source[args.source_modality].to(device=device, dtype=torch.float32)
            source_label = maybe_squeeze_label(batch_source["seg"].to(device=device, dtype=torch.long))
            target_image = batch_target[args.target_modality].to(device=device, dtype=torch.float32)

            source_feature_code = se_source(source_image)
            source_logits = sd(source_feature_code)
            source_feature_map = sd(source_feature_code, Seg_D2=True)
            loss_seg_source = criterion_ce(source_logits, source_label)

            build_source_bank_from_batch(
                source_feature_map=source_feature_map,
                source_label=source_label,
                prototype_bank=prototype_bank,
                num_classes=args.num_classes,
                boundary_kernel=args.boundary_kernel,
                neighbor_kernel=args.neighbor_kernel,
                keep_ratio=args.keep_ratio,
                min_points=args.min_boundary_points,
            )

            with torch.no_grad():
                teacher_target_code = teacher_se_target(target_image)
                teacher_target_logits = teacher_sd(teacher_target_code)
                teacher_target_feature_map = teacher_sd(teacher_target_code, Seg_D2=True)
                teacher_prob = torch.softmax(teacher_target_logits, dim=1)
                teacher_pseudo_label = torch.argmax(teacher_prob, dim=1)

                prompt_result_list = build_target_prompt_results(
                    teacher_feature_map=teacher_target_feature_map,
                    teacher_prob_map=teacher_prob,
                    target_pseudo_label=teacher_pseudo_label,
                    prototype_bank=prototype_bank,
                    boundary_kernel=args.boundary_kernel,
                    neighbor_kernel=args.neighbor_kernel,
                    box_padding=args.box_padding,
                    segments_per_strip=args.segments_per_strip,
                    prompt_min_score=args.prompt_min_score,
                )

                if len(prompt_result_list) > 0:
                    refined_target_label = refine_target_pseudo_labels_with_sam(
                        predictor=sam_predictor,
                        target_images=target_image,
                        target_pseudo_label=teacher_pseudo_label,
                        prompt_result_list=prompt_result_list,
                    )
                else:
                    refined_target_label = teacher_pseudo_label

            target_feature_code = se_target(target_image)
            target_logits = sd(target_feature_code)
            loss_seg_target = criterion_ce(target_logits, refined_target_label)

            total_loss = (
                float(args.source_loss_weight) * loss_seg_source
                + float(args.target_loss_weight) * loss_seg_target
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            update_ema_variables(se_target, teacher_se_target, alpha=float(args.ema_alpha))
            update_ema_variables(sd, teacher_sd, alpha=float(args.ema_alpha))

            train_iterator.set_postfix(
                total=float(total_loss.item()),
                src=float(loss_seg_source.item()),
                tgt=float(loss_seg_target.item()),
                prompts=len(prompt_result_list),
            )

        if (epoch + 1) % int(args.save_every) == 0:
            save_checkpoint(
                output_dir=output_dir,
                epoch=epoch,
                se_source=se_source,
                se_target=se_target,
                sd=sd,
                teacher_se_target=teacher_se_target,
                teacher_sd=teacher_sd,
            )


if __name__ == "__main__":
    main()
