from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from source_fine_boundary_points import (
    BoundaryKey,
    TensorLike,
    _create_demo_feature_map,
    _create_demo_seg_label,
    _print_demo_summary,
    build_source_fine_boundary_points,
)


def _subsample_features_for_visualization(
    features: torch.Tensor,
    max_points: int,
) -> torch.Tensor:
    """
    为了让特征可视化别把图挤炸，这里对每类点做一个轻量下采样。

    参数:
        features: torch.Tensor
            特征张量，形状为 `(N, C)`。
        max_points: int
            最多保留多少个点用于绘图。

    返回:
        sampled_features: torch.Tensor
            下采样后的特征张量。
    """
    if features.shape[0] <= max_points:
        return features

    # 这里直接随机采样就够了，目的是看整体分布，不是在做严格统计推断。
    indices = torch.randperm(features.shape[0], device=features.device)[:max_points]
    return features.index_select(0, indices)


def _compute_pca_projection_basis(
    features: torch.Tensor,
    n_components: int = 2,
):
    """
    基于输入特征计算 PCA 投影基。

    参数:
        features: torch.Tensor
            输入特征，形状为 `(N, C)`。
        n_components: int
            目标降维维度，当前默认 2。

    返回:
        mean_vector: torch.Tensor
            特征均值，形状为 `(C,)`。
        basis: torch.Tensor
            PCA 主方向矩阵，形状为 `(C, n_components)`。
    """
    if features.ndim != 2:
        raise ValueError(f"features 必须是二维张量，当前 shape={tuple(features.shape)}")
    if features.shape[0] == 0:
        raise ValueError("features 不能为空")

    mean_vector = features.mean(dim=0, keepdim=False)
    centered = features - mean_vector

    if features.shape[0] == 1:
        # 只有一个点时谈不上 PCA，这里退化成取前两个维度做占位投影。
        basis = torch.zeros((features.shape[1], n_components), device=features.device, dtype=features.dtype)
        basis[: min(features.shape[1], n_components), : min(features.shape[1], n_components)] = torch.eye(
            min(features.shape[1], n_components),
            device=features.device,
            dtype=features.dtype,
        )
        return mean_vector, basis

    # 协方差矩阵规模是 `(C, C)`，通道数通常远小于像素数，开销可接受。
    covariance = centered.t().matmul(centered) / max(features.shape[0] - 1, 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, -n_components:]
    return mean_vector, basis


def _project_features_with_basis(
    features: torch.Tensor,
    mean_vector: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """
    使用给定 PCA 均值和投影基，将特征投影到二维平面。

    参数:
        features: torch.Tensor
            输入特征，形状为 `(N, C)`。
        mean_vector: torch.Tensor
            均值向量，形状为 `(C,)`。
        basis: torch.Tensor
            投影基，形状为 `(C, 2)`。

    返回:
        projected: torch.Tensor
            二维投影结果，形状为 `(N, 2)`。
    """
    return (features - mean_vector).matmul(basis)


def visualize_boundary_features_before_after(
    raw_boundary_dict: Dict[BoundaryKey, Dict[str, torch.Tensor]],
    filtered_boundary_dict: Dict[BoundaryKey, Dict[str, torch.Tensor]],
    save_path: Union[str, Path],
    max_points_per_class: int = 1500,
    dpi: int = 220,
) -> None:
    """
    将过滤前后的边界点特征分布画成二维散点图，便于直观看筛选是否把离群点压掉了。

    可视化策略:
        1. 使用所有“原始边界点归一化特征 + 筛选后归一化特征”共同拟合一个二维 PCA
        2. 左图展示过滤前，右图展示过滤后
        3. 相同边界类别在左右图中使用相同颜色，便于直接对照

    参数:
        raw_boundary_dict: dict
            原始几何候选边界字典，要求其中已经包含 `features_norm`。
        filtered_boundary_dict: dict
            筛选后的边界字典，要求其中包含 `features_norm`。
        save_path: str 或 Path
            图像保存路径。
        max_points_per_class: int
            每个边界类别最多用于绘图的点数，避免类别特别大时把图挤满。
        dpi: int
            保存分辨率。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pair_keys = sorted(set(raw_boundary_dict.keys()) | set(filtered_boundary_dict.keys()))
    if len(pair_keys) == 0:
        raise ValueError("没有任何边界类别可供可视化")

    raw_features_for_pca: List[torch.Tensor] = []
    filtered_features_for_pca: List[torch.Tensor] = []

    for pair_key in pair_keys:
        if pair_key in raw_boundary_dict and "features_norm" in raw_boundary_dict[pair_key]:
            raw_features_for_pca.append(
                _subsample_features_for_visualization(
                    raw_boundary_dict[pair_key]["features_norm"],
                    max_points=max_points_per_class,
                )
            )
        if pair_key in filtered_boundary_dict and "features_norm" in filtered_boundary_dict[pair_key]:
            filtered_features_for_pca.append(
                _subsample_features_for_visualization(
                    filtered_boundary_dict[pair_key]["features_norm"],
                    max_points=max_points_per_class,
                )
            )

    combined_features = raw_features_for_pca + filtered_features_for_pca
    if len(combined_features) == 0:
        raise ValueError("raw_boundary_dict 和 filtered_boundary_dict 中都没有可视化特征")

    combined_features_tensor = torch.cat(combined_features, dim=0).detach().cpu().to(torch.float32)
    mean_vector, basis = _compute_pca_projection_basis(combined_features_tensor, n_components=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    color_map = plt.get_cmap("tab20", max(len(pair_keys), 1))

    for idx, pair_key in enumerate(pair_keys):
        color = color_map(idx)

        if pair_key in raw_boundary_dict and "features_norm" in raw_boundary_dict[pair_key]:
            raw_features = _subsample_features_for_visualization(
                raw_boundary_dict[pair_key]["features_norm"].detach().cpu().to(torch.float32),
                max_points=max_points_per_class,
            )
            raw_projected = _project_features_with_basis(raw_features, mean_vector, basis)
            axes[0].scatter(
                raw_projected[:, 0].numpy(),
                raw_projected[:, 1].numpy(),
                s=8,
                alpha=0.55,
                c=[color],
                label=str(pair_key),
                edgecolors="none",
            )

        if pair_key in filtered_boundary_dict and "features_norm" in filtered_boundary_dict[pair_key]:
            filtered_features = _subsample_features_for_visualization(
                filtered_boundary_dict[pair_key]["features_norm"].detach().cpu().to(torch.float32),
                max_points=max_points_per_class,
            )
            filtered_projected = _project_features_with_basis(filtered_features, mean_vector, basis)
            axes[1].scatter(
                filtered_projected[:, 0].numpy(),
                filtered_projected[:, 1].numpy(),
                s=8,
                alpha=0.65,
                c=[color],
                label=str(pair_key),
                edgecolors="none",
            )

    axes[0].set_title("Boundary Features Before Filtering")
    axes[1].set_title("Boundary Features After Filtering")
    for ax in axes:
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        ax.grid(alpha=0.2, linestyle="--")
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def visualize_boundary_points_on_label(
    seg_label_2d: TensorLike,
    boundary_points_dict: Dict[BoundaryKey, Dict[str, torch.Tensor]],
    save_path: Union[str, Path],
    target_batch_idx: int = 0,
    point_size: int = 10,
    dpi: int = 200,
) -> None:
    """
    将不同细粒度边界类别的点可视化叠加到标签图上，并保存结果图片。

    参数:
        seg_label_2d: TensorLike
            单张 2D 标签图，形状为 `(H, W)`。
        boundary_points_dict: dict
            边界点字典，通常传 `filtered_boundary_dict` 或 `raw_boundary_dict`。
        save_path: str 或 Path
            可视化保存路径。
        target_batch_idx: int
            若输入字典中包含多 batch 坐标，则只可视化该 batch 的点。
        point_size: int
            散点尺寸。
        dpi: int
            图像保存分辨率。
    """
    if isinstance(seg_label_2d, torch.Tensor):
        seg_label_np = seg_label_2d.detach().cpu().numpy()
    else:
        seg_label_np = np.asarray(seg_label_2d)

    if seg_label_np.ndim != 2:
        raise ValueError(f"seg_label_2d 必须是二维数组，当前 shape={seg_label_np.shape}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(seg_label_np, cmap="tab20", interpolation="nearest", alpha=0.75)

    pair_keys = sorted(boundary_points_dict.keys())
    point_cmap = plt.get_cmap("gist_ncar", max(len(pair_keys), 1))

    for idx, pair_key in enumerate(pair_keys):
        info = boundary_points_dict[pair_key]
        coords = info["coords"].detach().cpu()
        if coords.numel() == 0:
            continue

        # 这里只显示指定 batch 的点，避免多张图混在一起看得头疼。
        batch_mask = coords[:, 0] == target_batch_idx
        coords_batch = coords[batch_mask]
        if coords_batch.numel() == 0:
            continue

        y = coords_batch[:, 1].numpy()
        x = coords_batch[:, 2].numpy()
        ax.scatter(
            x,
            y,
            s=point_size,
            c=[point_cmap(idx)],
            label=str(pair_key),
            marker="o",
            edgecolors="none",
        )

    ax.set_title("Fine Boundary Points Overlay")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8, ncol=1)
    ax.set_xlim(0, seg_label_np.shape[1] - 1)
    ax.set_ylim(seg_label_np.shape[0] - 1, 0)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    """
    最小可运行 demo。

    demo 内容:
        1. 构造一个包含背景和 3 个前景类别的二维标签图
        2. 构造一个与标签对齐的模拟特征图
        3. 跑通细粒度边界点提取与特征一致性筛选
        4. 打印每个边界类别筛选前后的数量
        5. 保存一张可视化图像
    """
    torch.manual_seed(42)
    np.random.seed(42)

    seg_label = _create_demo_seg_label(height=128, width=128)
    feature_map = _create_demo_feature_map(seg_label=seg_label, channels=16)

    results = build_source_fine_boundary_points(
        feature_map=feature_map,
        seg_label=seg_label,
        num_classes=4,
        boundary_kernel=3,
        keep_ratio=0.8,
        min_points=10,
        ignore_background_order=True,
    )

    _print_demo_summary(results["summary_info"])

    save_path = Path(__file__).resolve().parent / "demo_source_fine_boundary_points.png"
    visualize_boundary_points_on_label(
        seg_label_2d=seg_label[0],
        boundary_points_dict=results["filtered_boundary_dict"],
        save_path=save_path,
        target_batch_idx=0,
        point_size=12,
        dpi=200,
    )

    feature_save_path = Path(__file__).resolve().parent / "demo_source_fine_boundary_features.png"
    visualize_boundary_features_before_after(
        raw_boundary_dict=results["raw_boundary_dict"],
        filtered_boundary_dict=results["filtered_boundary_dict"],
        save_path=feature_save_path,
        max_points_per_class=1200,
        dpi=220,
    )

    print("-" * 80)
    print(f"可视化结果已保存到: {save_path}")
    print(f"特征可视化结果已保存到: {feature_save_path}")
    print("-" * 80)


if __name__ == "__main__":
    main()
