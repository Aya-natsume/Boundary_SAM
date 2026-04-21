"""PICAI 数据集与切片增强实现。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import gaussian_filter, zoom
from torch.utils.data import Dataset


def _to_path(path_like: str | Path) -> Path:
    """把输入统一转成 Path，少一点路径分支判断。"""
    return Path(path_like).expanduser().resolve()


def random_rot_flip(image: np.ndarray, label: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """随机旋转再翻转，算是最常见也最稳定的二维增强。"""
    rotate_k = np.random.randint(0, 4)
    flip_axis = np.random.randint(0, 2)
    image = np.rot90(image, rotate_k)
    image = np.flip(image, axis=flip_axis).copy()
    if label is None:
        return image, None
    label = np.rot90(label, rotate_k)
    label = np.flip(label, axis=flip_axis).copy()
    return image, label


def random_rotate(image: np.ndarray, label: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """随机小角度旋转，增强视角扰动。"""
    angle = np.random.randint(-20, 21)
    image = ndimage.rotate(image, angle, order=1, reshape=False)
    if label is None:
        return image, None
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def gamma_correction(image: np.ndarray) -> np.ndarray:
    """随机 gamma 变换，轻度扰动强度分布。"""
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max <= image_min:
        return image
    gamma = float(np.random.uniform(0.7, 1.5))
    image = (image - image_min) / (image_max - image_min)
    if np.random.rand() < 0.5:
        image = 1.0 - image
    image = np.power(image, gamma)
    image = image * (image_max - image_min) + image_min
    return image


def contrast_augment(image: np.ndarray) -> np.ndarray:
    """轻度对比度增强，保持上下界不跑飞。"""
    image_mean = float(image.mean())
    image_min = float(image.min())
    image_max = float(image.max())
    factor = float(np.random.uniform(0.9, 1.1))
    image = (image - image_mean) * factor + image_mean
    image = np.clip(image, image_min, image_max)
    return image


def gaussian_blur(image: np.ndarray) -> np.ndarray:
    """轻度高斯模糊，模拟采样和重建差异。"""
    sigma = float(np.random.uniform(0.0, 1.0))
    return gaussian_filter(image, sigma=sigma, order=0)


def gaussian_noise(image: np.ndarray) -> np.ndarray:
    """加一点高斯噪声，别把网络宠坏。"""
    noise = np.random.normal(loc=0.0, scale=0.05, size=image.shape)
    return image + noise


class RandomSliceTransform:
    """面向二维切片的增强器，接口和旧版 RandomGenerator 保持兼容。"""

    def __init__(
        self,
        output_size: Tuple[int, int],
        SpatialAug: bool = True,
        IntensityAug: bool = True,
        NonlinearAug: bool = False,
    ) -> None:
        self.output_size = tuple(output_size)
        self.spatial_aug = SpatialAug
        self.intensity_aug = IntensityAug
        self.nonlinear_aug = NonlinearAug

    def _resize(self, image: np.ndarray, label: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """把切片缩放到目标大小。"""
        height, width = image.shape
        target_h, target_w = self.output_size
        image = zoom(image, (target_h / height, target_w / width), order=1)
        if label is None:
            return image, None
        label = zoom(label, (target_h / height, target_w / width), order=0)
        return image, label

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """执行增强，并把结果转成训练可直接使用的 tensor。"""
        image = sample["image"]
        label = sample.get("label")

        if self.nonlinear_aug and np.random.rand() > 0.5:
            image = gamma_correction(image)

        if self.spatial_aug:
            if np.random.rand() > 0.5:
                image, label = random_rot_flip(image, label)
            elif np.random.rand() > 0.5:
                image, label = random_rotate(image, label)

        if self.intensity_aug:
            if np.random.rand() > 0.7:
                image = gamma_correction(image)
            if np.random.rand() > 0.7:
                image = contrast_augment(image)
            if np.random.rand() > 0.7:
                image = gaussian_blur(image)
            if np.random.rand() > 0.5:
                image = gaussian_noise(image)

        image, label = self._resize(image, label)
        image_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        result = {"image": image_tensor}
        if label is not None:
            result["label"] = torch.from_numpy(label.astype(np.int64))
        return result


class PICAIUnpairedSliceDataset(Dataset):
    """PICAI 训练集，按病人内连续切片窗口读取。"""

    def __init__(
        self,
        data_root: str | Path,
        modality: str,
        n_slices: int,
        transform: Optional[RandomSliceTransform] = None,
    ) -> None:
        self.data_root = _to_path(data_root)
        self.modality = modality.lower()
        self.n_slices = int(n_slices)
        self.transform = transform
        if self.modality not in {"t2w", "adc"}:
            raise ValueError(f"Unsupported modality: {modality}")
        if self.n_slices < 1:
            raise ValueError("n_slices must be >= 1")

        self.file_path = self.data_root / f"unpaired_{self.modality}.h5"
        self._file: Optional[h5py.File] = None

        with h5py.File(self.file_path, "r") as data_file:
            self.patient_count = int(data_file[self.modality].shape[0])
            self.slice_count = int(data_file[self.modality].shape[1])
            self.has_label = "seg" in data_file

        self.window_count = self.slice_count - self.n_slices + 1
        if self.window_count < 1:
            raise ValueError("n_slices is larger than the available slice count")
        self.length = self.patient_count * self.window_count

    def _get_file(self) -> h5py.File:
        """按需打开 h5 文件。"""
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def _apply_transform(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """把增强逻辑单独收起来，主流程会清楚很多。"""
        if self.transform is None:
            image_tensor = torch.from_numpy(image.astype(np.float32))
            label_tensor = None if label is None else torch.from_numpy(label.astype(np.int64))
            return image_tensor, label_tensor

        if label is None:
            image_list = []
            for slice_index in range(self.n_slices):
                dummy_label = np.zeros_like(image[slice_index], dtype=np.int64)
                transformed = self.transform({"image": image[slice_index], "label": dummy_label})
                image_list.append(transformed["image"].float())
            if self.n_slices == 1:
                return image_list[0], None
            image_tensor = torch.stack(image_list, dim=0).squeeze(1)
            return image_tensor, None

        if self.n_slices == 1:
            transformed = self.transform({"image": image[0], "label": label[0]})
            return transformed["image"].float(), transformed["label"].long()

        image_list = []
        label_list = []
        for slice_index in range(self.n_slices):
            transformed = self.transform({"image": image[slice_index], "label": label[slice_index]})
            image_list.append(transformed["image"].float())
            label_list.append(transformed["label"].long())

        image_tensor = torch.stack(image_list, dim=0).squeeze(1)
        label_tensor = torch.stack(label_list, dim=0)
        return image_tensor, label_tensor

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        data_file = self._get_file()
        patient_index = index // self.window_count
        slice_index = index % self.window_count

        image = data_file[self.modality][patient_index, slice_index:slice_index + self.n_slices]
        label = None
        if self.has_label:
            label = data_file["seg"][patient_index, slice_index:slice_index + self.n_slices]

        image_tensor, label_tensor = self._apply_transform(image=image, label=label)
        item = {self.modality: image_tensor}
        if label_tensor is not None:
            item["seg"] = label_tensor
        return item

    def __len__(self) -> int:
        return self.length

    def __del__(self) -> None:
        if getattr(self, "_file", None) is not None:
            try:
                self._file.close()
            except Exception:
                pass


class PICAIPairedVolumeDataset(Dataset):
    """PICAI 验证/测试集，按整例 volume 读取。"""

    def __init__(
        self,
        data_root: str | Path,
        source_modality: str,
        target_modality: str,
        image_transform=None,
    ) -> None:
        self.data_root = _to_path(data_root)
        self.source_modality = source_modality.lower()
        self.target_modality = target_modality.lower()
        self.image_transform = image_transform
        self.file_path = self._resolve_file_path()
        self._file: Optional[h5py.File] = None

        with h5py.File(self.file_path, "r") as data_file:
            self.length = int(data_file[self.source_modality].shape[0])
            self.has_label = "seg" in data_file

    def _resolve_file_path(self) -> Path:
        """自动匹配 paired 文件名，别让命名顺序这种小事卡住实验。"""
        candidates = [
            self.data_root / f"paired_{self.target_modality}_{self.source_modality}.h5",
            self.data_root / f"paired_{self.source_modality}_{self.target_modality}.h5",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Cannot find paired PICAI file under {self.data_root}")

    def _get_file(self) -> h5py.File:
        """按需打开 h5 文件。"""
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """把 numpy volume 转成 float tensor。"""
        if self.image_transform is not None:
            return self.image_transform(array)
        return torch.from_numpy(array.astype(np.float32))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        data_file = self._get_file()
        source_image = self._to_tensor(data_file[self.source_modality][index])
        target_image = self._to_tensor(data_file[self.target_modality][index])
        item = {self.source_modality: source_image, self.target_modality: target_image}
        if self.has_label:
            item["seg"] = torch.from_numpy(data_file["seg"][index].astype(np.int64))
        return item

    def __len__(self) -> int:
        return self.length

    def __del__(self) -> None:
        if getattr(self, "_file", None) is not None:
            try:
                self._file.close()
            except Exception:
                pass


class PICAITrainDataset_with_transform(PICAIUnpairedSliceDataset):
    """兼容旧训练脚本的类名，接口不变。"""


class PICAITrainDataset(PICAIUnpairedSliceDataset):
    """兼容旧训练脚本的类名，接口不变。"""

    def __init__(self, data_root: str | Path, modality: str, n_slices: int) -> None:
        super().__init__(data_root=data_root, modality=modality, n_slices=n_slices, transform=None)


class PICAITestDataset(PICAIPairedVolumeDataset):
    """兼容旧训练脚本的类名，接口不变。"""


RandomGenerator = RandomSliceTransform

