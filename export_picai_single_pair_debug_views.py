"""导出 PICAI 单病例单 pair 的详细 point-prompt 调试图。"""

from __future__ import annotations

from pathlib import Path

import torch

from ordered_boundary_point_prompt_generation import visualize_point_prompts_in_box
from visualize_source_fine_boundary_points_picai_pretrained import (
    PICAITrainDataset,
    build_boundary_prototype_bank_from_reference_subset,
    build_ordered_point_prompt_visualization_case,
    choose_slice_index_from_seg_volume,
    load_reference_source_models,
)


def export_single_pair_debug_views(save_dir: str | Path) -> None:
    """复用现有 PICAI 流程，导出每个 pair 的详细调试图。"""
    save_dir = Path(save_dir).expanduser().resolve()
    pseudo_dir = save_dir / "pseudo"
    gt_dir = save_dir / "gt"
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    se_source, seg_decoder = load_reference_source_models(device=device)
    bank = build_boundary_prototype_bank_from_reference_subset(
        device=device,
        se_source=se_source,
        seg_decoder=seg_decoder,
        patient_indices=(0, 1, 2, 3),
        slice_ranks=(0, 1),
        boundary_kernel=5,
        neighbor_kernel=3,
        keep_ratio=0.8,
        min_points=10,
    )

    dataset = PICAITrainDataset(
        data_root=str(Path("/home/chenxu/dataset/picai").expanduser().resolve() / "train"),
        modality="t2w",
        n_slices=1,
    )

    for patient_index in (0, 1, 2, 3):
        slice_index = choose_slice_index_from_seg_volume(
            dataset.data_file["seg"][int(patient_index)],
            top_rank=0,
        )
        case_record = build_ordered_point_prompt_visualization_case(
            device=device,
            se_source=se_source,
            seg_decoder=seg_decoder,
            bank=bank,
            patient_index=int(patient_index),
            slice_index=int(slice_index),
            boundary_kernel=5,
            neighbor_kernel=3,
            box_padding=6,
            use_soft_uncertainty=False,
            offset_distance=3.0,
            window_radius=12,
            sigma=5.0,
            topk_per_side=3,
            min_distance=5.0,
            min_score=None,
        )

        for geometry_name, label_key, result_key, geometry_dir in (
            ("pseudo", "pseudo_label", "pseudo_point_prompt_results", pseudo_dir),
            ("gt", "gt_label", "gt_point_prompt_results", gt_dir),
        ):
            background_2d = case_record[label_key]
            for result in case_record[result_key]:
                a = int(result["a"])
                b = int(result["b"])
                output_path = geometry_dir / (
                    f"patient_{int(case_record['patient_index']):03d}"
                    f"_slice_{int(case_record['slice_index']):02d}"
                    f"_pair_{a}_{b}_{geometry_name}.png"
                )
                # 这里直接复用第四部分自带的详细调试图，把中心点、法线、锚点和分层点全部画出来。
                visualize_point_prompts_in_box(
                    image_or_mask_2d=background_2d,
                    box=result["box"],
                    core_ab=result["core_ab"],
                    core_ba=result["core_ba"],
                    ref_center_a=result["ref_center_a"],
                    ref_center_b=result["ref_center_b"],
                    points_a=result["points_a"],
                    points_b=result["points_b"],
                    a=a,
                    b=b,
                    save_path=output_path,
                    center_point=result.get("center_point"),
                    normal_vector=result.get("normal_vector"),
                    tangent_vector=result.get("tangent_vector"),
                    boundary_points_global=result.get("boundary_points_global"),
                    anchor_a=result.get("anchor_a"),
                    anchor_b=result.get("anchor_b"),
                )
                print(
                    f"[{geometry_name}] patient={int(case_record['patient_index']):03d} "
                    f"slice={int(case_record['slice_index']):02d} "
                    f"pair=({a},{b}) -> {output_path}"
                )


def main() -> None:
    """脚本入口。"""
    export_single_pair_debug_views(
        save_dir=Path(__file__).resolve().parent / "outputs" / "picai_single_pair_debug_views"
    )


if __name__ == "__main__":
    main()
