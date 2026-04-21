"""Boundary_SAM 的简化测试脚本。

测试目标：
1. 加载训练好的目标域编码器和共享解码器。
2. 在 PICAI 测试集的目标域 volume 上逐 slice 推理。
3. 输出每个前景类别的 Dice 和整体均值，并可选保存预测体积。

说明：
1. 这里保持最小实现，只评估学生模型本身，不再把 SAM 放进测试环节。
2. 训练时 SAM 的职责是生成更干净的目标域伪标签；测试时一般直接看学生模型分割效果。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.Seg import Decoder, Encoder, load_pretrained_weights
from picai_dataset import PICAITestDataset


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Test Boundary_SAM student model on PICAI.")
    parser.add_argument("--data-root", type=str, default="/home/chenxu/dataset/picai")
    parser.add_argument("--source-modality", type=str, default="t2w")
    parser.add_argument("--target-modality", type=str, default="adc")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--infer-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-encoder-checkpoint", type=str, required=True)
    parser.add_argument("--decoder-checkpoint", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs" / "test"))
    parser.add_argument("--save-prediction", action="store_true")
    return parser.parse_args()


def maybe_import_nibabel():
    """按需导入 nibabel。"""
    try:
        import nibabel as nib  # type: ignore
    except Exception:
        return None
    return nib


def compute_case_dice(pred_label: torch.Tensor, gt_label: torch.Tensor, num_classes: int) -> List[float]:
    """计算单个 volume 的前景 Dice。"""
    dice_list: List[float] = []
    for class_index in range(1, int(num_classes)):
        pred_mask = pred_label == int(class_index)
        gt_mask = gt_label == int(class_index)
        intersection = torch.logical_and(pred_mask, gt_mask).sum().item()
        pred_sum = pred_mask.sum().item()
        gt_sum = gt_mask.sum().item()
        if pred_sum + gt_sum == 0:
            dice = 1.0
        else:
            dice = 2.0 * float(intersection) / float(pred_sum + gt_sum)
        dice_list.append(dice)
    return dice_list


def save_prediction_volume(save_dir: Path, case_index: int, pred_volume: np.ndarray) -> None:
    """保存预测体积。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    nib = maybe_import_nibabel()
    if nib is not None:
        nib.save(
            nib.Nifti1Image(pred_volume.astype(np.uint16), np.eye(4)),
            str(save_dir / f"case_{case_index:03d}.nii.gz"),
        )
        return
    np.save(save_dir / f"case_{case_index:03d}.npy", pred_volume.astype(np.uint16))


def main() -> None:
    """测试入口。"""
    args = parse_args()
    save_dir = Path(args.save_dir).expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    se_target = Encoder().to(device)
    sd = Decoder(num_class=args.num_classes).to(device)
    load_pretrained_weights(se_target, args.target_encoder_checkpoint, strict=True, map_location=device)
    load_pretrained_weights(sd, args.decoder_checkpoint, strict=True, map_location=device)
    se_target.eval()
    sd.eval()

    test_loader = DataLoader(
        dataset=PICAITestDataset(
            data_root=Path(args.data_root) / "test",
            source_modality=args.target_modality,
            target_modality=args.source_modality,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    all_case_dice: List[List[float]] = []
    with torch.no_grad():
        for case_index, batch in enumerate(test_loader):
            target_volume = batch[args.target_modality].to(device=device, dtype=torch.float32)
            gt_volume = batch["seg"].to(device=device, dtype=torch.long)

            # 当前网络是 2D 分割器，所以测试时按 slice 维度摊平成一个大 batch 推理。
            if target_volume.dim() != 4:
                raise ValueError("Expected target volume with shape (B, D, H, W)")
            if target_volume.size(0) != 1:
                raise ValueError("This simple test script expects batch_size=1 for volume inference")

            slice_tensor = target_volume[0].unsqueeze(1)
            logits_list: List[torch.Tensor] = []
            for start_index in range(0, int(slice_tensor.size(0)), int(args.infer_batch_size)):
                end_index = min(int(slice_tensor.size(0)), start_index + int(args.infer_batch_size))
                slice_batch = slice_tensor[start_index:end_index]
                logits_list.append(sd(se_target(slice_batch)))
            logits_volume = torch.cat(logits_list, dim=0)
            pred_volume = torch.argmax(logits_volume, dim=1)

            case_dice = compute_case_dice(
                pred_label=pred_volume,
                gt_label=gt_volume[0],
                num_classes=args.num_classes,
            )
            all_case_dice.append(case_dice)

            if args.save_prediction:
                save_prediction_volume(
                    save_dir=save_dir,
                    case_index=case_index,
                    pred_volume=pred_volume.cpu().numpy(),
                )

            dice_text = " | ".join(
                [f"class_{class_index + 1}_dice={score:.4f}" for class_index, score in enumerate(case_dice)]
            )
            print(f"case={case_index:03d} | {dice_text}")

    dice_array = np.asarray(all_case_dice, dtype=np.float32)
    class_mean = dice_array.mean(axis=0)
    for class_index, score in enumerate(class_mean, start=1):
        print(f"class_{class_index}_mean_dice={float(score):.4f}")
    print(f"foreground_mean_dice={float(class_mean.mean()):.4f}")


if __name__ == "__main__":
    main()
