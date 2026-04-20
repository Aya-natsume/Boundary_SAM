from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

# 项目里有些旧脚本用了“同目录裸导入”，这里先把路径垫好，不然一到真实 runner 就容易绊一下。
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from .model.Seg import Decoder, Encoder
    from .model.build import load_checkpoint_state_dict, load_pretrained_weights
    from .source_boundary_single_prototype_library import (
        build_source_boundary_prototype_library,
        inspect_boundary_prototype_library,
    )
    from .source_fine_boundary_points import build_source_fine_boundary_points
    from .source_fine_boundary_visualization import (
        visualize_boundary_features_before_after,
        visualize_boundary_points_on_label,
    )
    from .target_boundary_prompt_score import generate_boundary_prompt_scores
except ImportError:
    from model.Seg import Decoder, Encoder
    from model.build import load_checkpoint_state_dict, load_pretrained_weights
    from source_boundary_single_prototype_library import (
        build_source_boundary_prototype_library,
        inspect_boundary_prototype_library,
    )
    from source_fine_boundary_points import build_source_fine_boundary_points
    from source_fine_boundary_visualization import (
        visualize_boundary_features_before_after,
        visualize_boundary_points_on_label,
    )
    from target_boundary_prompt_score import generate_boundary_prompt_scores


class PICAIH5SliceDataset(Dataset):
    """
    将 PICAI 的 H5 volume 数据包装成按 2D slice 访问的数据集。

    参数:
        h5_path: str
            H5 文件路径。
        image_key: str
            图像字段名，例如 `t2w` 或 `adc`。
        label_key: Optional[str]
            标签字段名，通常为 `seg`。
        selected_indices: Sequence[Tuple[int, int]]
            需要读取的 `(patient_idx, slice_idx)` 列表。

    返回样本格式:
        {
            "image": Tensor[1, H, W],
            "seg_label": Tensor[H, W],  # 如果有标签
            "patient_idx": int,
            "slice_idx": int,
        }
    """

    def __init__(
        self,
        h5_path: str,
        image_key: str,
        label_key: Optional[str],
        selected_indices: Sequence[Tuple[int, int]],
    ) -> None:
        self.h5_path = str(h5_path)
        self.image_key = str(image_key)
        self.label_key = str(label_key) if label_key is not None else None
        self.selected_indices = [(int(patient_idx), int(slice_idx)) for patient_idx, slice_idx in selected_indices]
        self._data_file: Optional[h5py.File] = None

    def _get_data_file(self) -> h5py.File:
        """
        延迟打开 H5 文件句柄，避免 DataLoader 初始化阶段过早占资源。
        """
        if self._data_file is None:
            self._data_file = h5py.File(self.h5_path, "r")
        return self._data_file

    def __len__(self) -> int:
        """
        返回数据集长度。
        """
        return len(self.selected_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        读取一个指定的 `(patient_idx, slice_idx)` 样本。
        """
        patient_idx, slice_idx = self.selected_indices[index]
        data_file = self._get_data_file()

        image = data_file[self.image_key][patient_idx, slice_idx].astype(np.float32)
        sample: Dict[str, Any] = {
            "image": torch.from_numpy(image).unsqueeze(0),
            "patient_idx": int(patient_idx),
            "slice_idx": int(slice_idx),
        }

        if self.label_key is not None and self.label_key in data_file:
            seg_label = data_file[self.label_key][patient_idx, slice_idx].astype(np.int64)
            sample["seg_label"] = torch.from_numpy(seg_label)

        return sample

    def __del__(self) -> None:
        """
        数据集销毁时关闭 H5 句柄。
        """
        if getattr(self, "_data_file", None) is not None:
            self._data_file.close()


class SegFeatureExtractor(nn.Module):
    """
    将编码器和解码器包装成只输出 segmentation feature map 的模块。

    输入:
        image: `(B, 1, H, W)`

    输出:
        {
            "feature_map": Tensor, shape = `(B, 32, H, W)`
        }
    """

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向提取全分辨率 segmentation feature map。
        """
        encoded_feature = self.encoder(image)
        # `Seg_D2=True` 时返回的是 block3 之后的归一化解码特征，分辨率已经回到输入尺度。
        feature_map = self.decoder(encoded_feature, Seg_D2=True)
        return {"feature_map": feature_map}


def _load_decoder_checkpoint_with_boundary_sam_compatibility(
    decoder: Decoder,
    checkpoint_path: str,
) -> None:
    """
    加载一个与 Boundary_SAM 解码器几乎兼容的 checkpoint。

    当前兼容处理:
        DADASeg 的最后一层 key 可能是 `block4.weight / block4.bias`，
        而 Boundary_SAM 这里写成了 `block4.0.weight / block4.0.bias`。
        这里做一次轻量映射，不然最后一层会闹别扭。
    """
    state_dict = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
    remapped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key == "block4.weight":
            remapped_state_dict["block4.0.weight"] = value
        elif key == "block4.bias":
            remapped_state_dict["block4.0.bias"] = value
        else:
            remapped_state_dict[key] = value
    decoder.load_state_dict(remapped_state_dict, strict=True)


def _select_source_positive_slices(
    h5_path: str,
    max_samples: int = 96,
    max_patients: int = 24,
) -> List[Tuple[int, int]]:
    """
    从源域 T2W 训练集里挑一批带前景的 2D slice，用来构建 prototype library。

    选择策略:
        1. 只保留 `seg.sum() > 0` 的切片
        2. 优先从前若干病人里均匀取样，避免全挤在一个病人上
        3. 最终截断到 `max_samples`
    """
    selected_indices: List[Tuple[int, int]] = []
    with h5py.File(h5_path, "r") as data_file:
        seg_volume = data_file["seg"]
        num_patients = min(int(seg_volume.shape[0]), int(max_patients))
        num_slices = int(seg_volume.shape[1])

        for patient_idx in range(num_patients):
            foreground_sizes: List[Tuple[int, int]] = []
            for slice_idx in range(num_slices):
                foreground_size = int((seg_volume[patient_idx, slice_idx] > 0).sum())
                if foreground_size > 0:
                    foreground_sizes.append((slice_idx, foreground_size))

            # 每个病人内部优先选前景面积更大的切片，这样边界和类别都会更完整一些。
            foreground_sizes.sort(key=lambda item: item[1], reverse=True)
            for slice_idx, _ in foreground_sizes[: max(1, max_samples // max(num_patients, 1)) + 2]:
                selected_indices.append((patient_idx, slice_idx))
                if len(selected_indices) >= max_samples:
                    return selected_indices

    return selected_indices[:max_samples]


def _select_target_representative_slices(
    h5_path: str,
    num_patients: int = 3,
) -> List[Tuple[int, int]]:
    """
    从目标域 ADC 测试集里为前几个病人各挑一张前景最明显的 slice。

    这样保存出来的图一般会更像样，不至于抽到几乎空白的外围切片。
    """
    selected_indices: List[Tuple[int, int]] = []
    with h5py.File(h5_path, "r") as data_file:
        seg_volume = data_file["seg"]
        total_patients = min(int(seg_volume.shape[0]), int(num_patients))
        num_slices = int(seg_volume.shape[1])

        for patient_idx in range(total_patients):
            best_slice_idx = 0
            best_foreground_size = -1
            for slice_idx in range(num_slices):
                foreground_size = int((seg_volume[patient_idx, slice_idx] > 0).sum())
                if foreground_size > best_foreground_size:
                    best_foreground_size = foreground_size
                    best_slice_idx = slice_idx
            selected_indices.append((patient_idx, best_slice_idx))

    return selected_indices


def _build_boundary_point_fn() -> Any:
    """
    构造 source boundary point 提取函数包装器，统一固定超参数。
    """

    def boundary_point_fn(
        feature_map: torch.Tensor,
        seg_label: torch.Tensor,
        num_classes: int,
    ) -> Dict[str, Any]:
        return build_source_fine_boundary_points(
            feature_map=feature_map,
            seg_label=seg_label,
            num_classes=num_classes,
            boundary_kernel=3,
            keep_ratio=0.8,
            min_points=10,
            ignore_background_order=True,
        )

    return boundary_point_fn


def _save_target_prompt_overview(
    image_2d: torch.Tensor,
    gt_mask_2d: torch.Tensor,
    pred_mask_2d: torch.Tensor,
    score_maps_dict: Dict[int, torch.Tensor],
    prompt_seed_dict: Dict[int, List[Dict[str, Any]]],
    save_path: Path,
    patient_idx: int,
    slice_idx: int,
) -> None:
    """
    保存一张 target prompt score 总览图。

    图像布局:
        1. ADC 原图
        2. GT 标签
        3. 粗分割预测
        4. organ 1 score map
        5. organ 2 score map
        6. 粗分割 + 两个器官的 prompt seeds 叠加图
    """
    image_np = image_2d.detach().cpu().numpy()
    gt_np = gt_mask_2d.detach().cpu().numpy()
    pred_np = pred_mask_2d.detach().cpu().numpy()
    organ_ids = sorted(score_maps_dict.keys())

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax_list = axes.flatten()

    ax_list[0].imshow(image_np, cmap="gray")
    ax_list[0].set_title(f"ADC Image\npatient={patient_idx}, slice={slice_idx}")
    ax_list[0].axis("off")

    ax_list[1].imshow(gt_np, cmap="tab20", interpolation="nearest")
    ax_list[1].set_title("GT Segmentation")
    ax_list[1].axis("off")

    ax_list[2].imshow(pred_np, cmap="tab20", interpolation="nearest")
    ax_list[2].set_title("Coarse Prediction")
    ax_list[2].axis("off")

    for score_plot_idx, organ_id in enumerate(organ_ids[:2], start=3):
        score_map_2d = score_maps_dict[organ_id][0].detach().cpu().numpy()
        im = ax_list[score_plot_idx].imshow(score_map_2d, cmap="magma")
        ax_list[score_plot_idx].set_title(
            f"Organ {organ_id} Score Map\nnum_seeds={len(prompt_seed_dict.get(organ_id, []))}"
        )
        ax_list[score_plot_idx].axis("off")
        fig.colorbar(im, ax=ax_list[score_plot_idx], fraction=0.046, pad=0.04)

    overlay_ax = ax_list[5]
    overlay_ax.imshow(pred_np, cmap="tab20", interpolation="nearest", alpha=0.80)
    overlay_ax.set_title("Prediction + Prompt Seeds")
    overlay_ax.axis("off")

    organ_colors = {1: "cyan", 2: "lime", 3: "orange"}
    for organ_id, seed_list in prompt_seed_dict.items():
        color = organ_colors.get(int(organ_id), "red")
        for point_idx, seed_info in enumerate(seed_list):
            batch_idx = int(seed_info["batch_idx"])
            if batch_idx != 0:
                continue

            y = int(seed_info["y"])
            x = int(seed_info["x"])
            score = float(seed_info["score"])
            conflict_class = int(seed_info["conflict_class"])
            overlay_ax.scatter(
                x,
                y,
                s=60,
                c=color,
                edgecolors="white",
                linewidths=1.0,
            )
            overlay_ax.text(
                x + 1.5,
                y - 1.5,
                f"{organ_id}:{conflict_class}\n{score:.3f}",
                color="white",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.2},
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def run_picai_real_module_check() -> Dict[str, Any]:
    """
    在真实 PICAI 数据上跑一遍 Boundary_SAM 当前四个核心功能模块。

    实测流程:
        1. 加载一个可复用的 PICAI 预训练 source / target encoder 和 decoder
        2. 用真实 T2W 训练切片构建 source prototype library
        3. 保存几张 source boundary point 可视化
        4. 用真实 ADC 测试切片生成 boundary prompt scores 和 seeds
        5. 保存几张 target prompt 可视化总览图

    说明:
        这里用的是工作区已有的 DADASeg PICAI checkpoint，
        只是借它提供一套真实 feature / coarse prediction，
        目的是检查 Boundary_SAM 这几个模块的空间行为是否合理。
    """
    torch.manual_seed(2026)

    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "picai_real_module_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_t2w_h5 = "/home/chenxu/dataset/picai/train/unpaired_t2w.h5"
    test_adc_h5 = "/home/chenxu/dataset/picai/test/paired_t2w_adc.h5"

    source_encoder_ckpt = (
        "/home/chenxu/Aya/project/DADASeg_picai_polished/outputs/"
        "picai_polished_fullrun/save/Seg_E_MRI_ema_11.pth"
    )
    target_encoder_ckpt = (
        "/home/chenxu/Aya/project/DADASeg_picai_polished/outputs/"
        "picai_polished_fullrun/save/Seg_E_CT_ema_11.pth"
    )
    decoder_ckpt = (
        "/home/chenxu/Aya/project/DADASeg_picai_polished/outputs/"
        "picai_polished_fullrun/save/Seg_D_ema_11.pth"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3

    # 这里直接复用 Boundary_SAM 自己的 Encoder/Decoder 定义，checkpoint 来自已训练好的 PICAI 模型。
    source_encoder = Encoder().to(device)
    target_encoder = Encoder().to(device)
    decoder = Decoder(num_class=num_classes).to(device)

    load_pretrained_weights(source_encoder, source_encoder_ckpt, strict=True)
    load_pretrained_weights(target_encoder, target_encoder_ckpt, strict=True)
    _load_decoder_checkpoint_with_boundary_sam_compatibility(decoder, decoder_ckpt)

    source_encoder.eval()
    target_encoder.eval()
    decoder.eval()

    source_feature_model = SegFeatureExtractor(source_encoder, decoder).to(device)
    boundary_point_fn = _build_boundary_point_fn()

    source_selected_indices = _select_source_positive_slices(
        h5_path=train_t2w_h5,
        max_samples=96,
        max_patients=24,
    )
    source_dataset = PICAIH5SliceDataset(
        h5_path=train_t2w_h5,
        image_key="t2w",
        label_key="seg",
        selected_indices=source_selected_indices,
    )
    source_dataloader = DataLoader(source_dataset, batch_size=8, shuffle=False, num_workers=0)

    print("=" * 100)
    print("Step 1/3: Build source boundary prototype library on PICAI T2W")
    print("=" * 100)
    prototype_library, summary_info = build_source_boundary_prototype_library(
        dataloader=source_dataloader,
        model=source_feature_model,
        num_classes=num_classes,
        device=device,
        boundary_point_fn=boundary_point_fn,
        save_path=output_dir / "picai_source_boundary_single_prototype_library.pt",
        ordered=False,
    )
    inspect_boundary_prototype_library(
        prototype_dict=prototype_library,
        summary_info=summary_info,
        min_img_count=4,
        min_point_count=80,
    )

    print("=" * 100)
    print("Step 2/3: Save source fine-boundary visualizations on real T2W slices")
    print("=" * 100)
    source_vis_indices = source_selected_indices[:2]
    source_vis_dataset = PICAIH5SliceDataset(
        h5_path=train_t2w_h5,
        image_key="t2w",
        label_key="seg",
        selected_indices=source_vis_indices,
    )

    with torch.no_grad():
        for vis_idx in range(len(source_vis_dataset)):
            sample = source_vis_dataset[vis_idx]
            image = sample["image"].unsqueeze(0).to(device)
            seg_label = sample["seg_label"].unsqueeze(0).to(device)
            patient_idx = int(sample["patient_idx"])
            slice_idx = int(sample["slice_idx"])

            encoded_feature = source_encoder(image)
            feature_map = decoder(encoded_feature, Seg_D2=True)
            boundary_results = build_source_fine_boundary_points(
                feature_map=feature_map,
                seg_label=seg_label,
                num_classes=num_classes,
                boundary_kernel=3,
                keep_ratio=0.8,
                min_points=10,
                ignore_background_order=True,
            )

            point_save_path = output_dir / f"source_patient_{patient_idx}_slice_{slice_idx}_boundary_points.png"
            visualize_boundary_points_on_label(
                seg_label_2d=seg_label[0].detach().cpu(),
                boundary_points_dict=boundary_results["filtered_boundary_dict"],
                save_path=point_save_path,
                target_batch_idx=0,
                point_size=14,
                dpi=220,
            )

            feature_save_path = output_dir / f"source_patient_{patient_idx}_slice_{slice_idx}_features_before_after.png"
            visualize_boundary_features_before_after(
                raw_boundary_dict=boundary_results["raw_boundary_dict"],
                filtered_boundary_dict=boundary_results["filtered_boundary_dict"],
                save_path=feature_save_path,
                max_points_per_class=1500,
                dpi=220,
            )

            print(f"saved source boundary visualization: {point_save_path}")
            print(f"saved source feature visualization: {feature_save_path}")

    print("=" * 100)
    print("Step 3/3: Generate target boundary prompt scores on real PICAI ADC slices")
    print("=" * 100)
    target_selected_indices = _select_target_representative_slices(
        h5_path=test_adc_h5,
        num_patients=3,
    )
    target_dataset = PICAIH5SliceDataset(
        h5_path=test_adc_h5,
        image_key="adc",
        label_key="seg",
        selected_indices=target_selected_indices,
    )

    target_result_paths: List[str] = []
    with torch.no_grad():
        for target_idx in range(len(target_dataset)):
            sample = target_dataset[target_idx]
            image = sample["image"].unsqueeze(0).to(device)
            seg_label = sample["seg_label"].unsqueeze(0).to(device)
            patient_idx = int(sample["patient_idx"])
            slice_idx = int(sample["slice_idx"])

            encoded_feature = target_encoder(image)
            feature_map = decoder(encoded_feature, Seg_D2=True)
            logits = decoder(encoded_feature, Seg_D2=False)
            prob_map = torch.softmax(logits, dim=1)
            pred_mask = prob_map.argmax(dim=1)

            score_maps_dict, prompt_seed_dict, _ = generate_boundary_prompt_scores(
                feature_map=feature_map,
                prob_map=prob_map,
                pred_mask=pred_mask,
                prototype_library=prototype_library,
                num_classes=num_classes,
                organ_ids=None,
                boundary_kernel=3,
                topk=10,
                min_score=0.0,
                min_distance=6,
                ordered=False,
                missing_proto_value=0.0,
            )

            for organ_id in sorted(prompt_seed_dict.keys()):
                print(
                    f"target patient={patient_idx}, slice={slice_idx}, organ={organ_id}, "
                    f"num_seeds={len(prompt_seed_dict[organ_id])}"
                )

            save_path = output_dir / f"target_patient_{patient_idx}_slice_{slice_idx}_prompt_overview.png"
            _save_target_prompt_overview(
                image_2d=image[0, 0].detach().cpu(),
                gt_mask_2d=seg_label[0].detach().cpu(),
                pred_mask_2d=pred_mask[0].detach().cpu(),
                score_maps_dict=score_maps_dict,
                prompt_seed_dict=prompt_seed_dict,
                save_path=save_path,
                patient_idx=patient_idx,
                slice_idx=slice_idx,
            )
            target_result_paths.append(str(save_path))
            print(f"saved target prompt overview: {save_path}")

    return {
        "output_dir": str(output_dir),
        "prototype_library_path": str(output_dir / "picai_source_boundary_single_prototype_library.pt"),
        "target_result_paths": target_result_paths,
        "source_selected_indices": source_selected_indices,
        "target_selected_indices": target_selected_indices,
    }


if __name__ == "__main__":
    run_picai_real_module_check()
