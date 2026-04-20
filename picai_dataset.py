import os
import random
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import gaussian_filter, zoom
from scipy.special import comb
from torch.utils.data import Dataset


def random_rot_flip(image: np.ndarray, label: Optional[np.ndarray] = None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()

    if label is None:
        return image

    label = np.rot90(label, k)
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image: np.ndarray, label: Optional[np.ndarray] = None, angle_range: Tuple[int, int] = (-20, 20)):
    angle = np.random.randint(angle_range[0], angle_range[1] + 1)
    image = ndimage.rotate(image, angle, order=1, reshape=False, mode="nearest")

    if label is None:
        return image

    label = ndimage.rotate(label, angle, order=0, reshape=False, mode="nearest")
    return image, label


def gaussian_noise(image: np.ndarray, label: Optional[np.ndarray] = None, std: float = 0.05):
    noise = np.random.normal(0.0, std, image.shape)
    image = image + noise
    if label is None:
        return image
    return image, label


def gaussian_blur(image: np.ndarray, label: Optional[np.ndarray] = None, sigma_range: Tuple[float, float] = (0.0, 1.0)):
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    image = gaussian_filter(image, sigma=sigma, order=0)
    if label is None:
        return image
    return image, label


def gamma_correction(image: np.ndarray, label: Optional[np.ndarray] = None, gamma_range: Tuple[float, float] = (0.7, 1.5)):
    gamma_value = random.random() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
    image_min = image.min()
    image_max = image.max()

    if image_min < image_max:
        image = (image - image_min) / (image_max - image_min)
        if random.random() < 0.5:
            image = 1.0 - image
        image = np.power(image, gamma_value) * (image_max - image_min) + image_min

    if label is None:
        return image
    return image, label


def contrast_augmentation(
    image: np.ndarray,
    label: Optional[np.ndarray] = None,
    contrast_range: Tuple[float, float] = (0.9, 1.1),
):
    factor = np.random.uniform(contrast_range[0], contrast_range[1])
    mean = image.mean()
    image_min = image.min()
    image_max = image.max()
    image = (image - mean) * factor + mean
    image = np.clip(image, image_min, image_max)

    if label is None:
        return image
    return image, label


def bernstein_poly(i: int, n: int, t: np.ndarray):
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points: Sequence[Sequence[float]], n_times: int = 1000):
    x_points = np.array([point[0] for point in points])
    y_points = np.array([point[1] for point in points])
    t = np.linspace(0.0, 1.0, n_times)
    polynomial_array = np.array([bernstein_poly(i, len(points) - 1, t) for i in range(len(points))])
    x_values = np.dot(x_points, polynomial_array)
    y_values = np.dot(y_points, polynomial_array)
    return x_values, y_values


def nonlinear_transformation(image: np.ndarray):
    points = [[0.0, 0.0], [random.random(), random.random()], [random.random(), random.random()], [1.0, 1.0]]
    x_values, y_values = bezier_curve(points, n_times=1000)

    if random.random() < 0.5:
        x_values = np.sort(x_values)
    else:
        x_values = np.sort(x_values)
        y_values = np.sort(y_values)

    return np.interp(image, x_values, y_values)


class RandomGenerator:
    """
    `augmentation_mode` controls how much augmentation is used:
    - `none`: only resize + tensor conversion
    - `basic`: spatial augmentation
    - `fine`: spatial + intensity + nonlinear augmentation
    """

    def __init__(self, output_size: Optional[Sequence[int]] = (192, 192), augmentation_mode: str = "none"):
        self.output_size = tuple(output_size) if output_size is not None else None
        self.augmentation_mode = augmentation_mode.lower()
        if self.augmentation_mode not in {"none", "basic", "fine"}:
            raise ValueError("augmentation_mode must be one of {'none', 'basic', 'fine'}")

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if self.output_size is None:
            return image
        if image.shape == tuple(self.output_size):
            return image

        scale = (self.output_size[0] / image.shape[0], self.output_size[1] / image.shape[1])
        return zoom(image, scale, order=1)

    def _resize_label(self, label: np.ndarray) -> np.ndarray:
        if self.output_size is None:
            return label
        if label.shape == tuple(self.output_size):
            return label

        scale = (self.output_size[0] / label.shape[0], self.output_size[1] / label.shape[1])
        return zoom(label, scale, order=0)

    def __call__(self, sample: Dict[str, np.ndarray]):
        image = sample["image"]
        label = sample.get("label")

        if self.augmentation_mode in {"basic", "fine"}:
            if random.random() > 0.5:
                if label is None:
                    image = random_rot_flip(image)
                else:
                    image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                if label is None:
                    image = random_rotate(image)
                else:
                    image, label = random_rotate(image, label)

        if self.augmentation_mode == "fine":
            if random.random() > 0.5:
                image = nonlinear_transformation(image)
            if random.random() > 0.7:
                if label is None:
                    image = gamma_correction(image)
                else:
                    image, label = gamma_correction(image, label)
            if random.random() > 0.7:
                if label is None:
                    image = contrast_augmentation(image)
                else:
                    image, label = contrast_augmentation(image, label)
            if random.random() > 0.7:
                if label is None:
                    image = gaussian_blur(image)
                else:
                    image, label = gaussian_blur(image, label)
            if random.random() > 0.5:
                if label is None:
                    image = gaussian_noise(image)
                else:
                    image, label = gaussian_noise(image, label)

        image = self._resize_image(image).astype(np.float32)
        image_tensor = torch.from_numpy(image).unsqueeze(0)

        output = {"image": image_tensor}
        if label is not None:
            label = self._resize_label(label).astype(np.int64)
            output["label"] = torch.from_numpy(label)

        return output


class PICAITrainDataset_with_transform(Dataset):
    def __init__(
        self,
        data_root: str,
        modality: str,
        n_slices: int,
        output_size: Optional[Sequence[int]] = (192, 192),
        augmentation_mode: str = "none",
        transform: Optional[RandomGenerator] = None,
    ):
        self.modality = modality.lower()
        if self.modality not in {"t2w", "adc"}:
            raise ValueError("modality must be 't2w' or 'adc'")

        if n_slices <= 0:
            raise ValueError("n_slices must be a positive integer")

        self.n_slices = n_slices
        self.output_size = tuple(output_size) if output_size is not None else None
        self.transform = transform if transform is not None else RandomGenerator(output_size, augmentation_mode)
        self.data_path = os.path.join(data_root, f"unpaired_{self.modality}.h5")
        self.data_file = None

        with h5py.File(self.data_path, "r") as data_file:
            self.patient_images = data_file[self.modality].shape[0]
            self.patient_slices = data_file[self.modality].shape[1]

        if self.n_slices > self.patient_slices:
            raise ValueError("n_slices must not exceed the number of slices per patient")

        self.slice_nums = self.patient_slices - self.n_slices + 1
        self.length = self.patient_images * self.slice_nums

    def _get_data_file(self):
        if self.data_file is None:
            self.data_file = h5py.File(self.data_path, "r")
        return self.data_file

    def __getitem__(self, index: int):
        data_file = self._get_data_file()
        image_index = index // self.slice_nums
        slice_index = index % self.slice_nums

        image = data_file[self.modality][image_index, slice_index:slice_index + self.n_slices]
        data_item = {}

        label_name = "seg"
        has_label = label_name in data_file
        label = None
        if has_label:
            label = data_file[label_name][image_index, slice_index:slice_index + self.n_slices].astype(np.int64)

        if self.n_slices == 1:
            sample = {"image": image[0]}
            if has_label:
                sample["label"] = label[0]
            sample = self.transform(sample)

            image_tensor = sample["image"].float()
            if image_tensor.ndim == 3 and image_tensor.shape[0] == 1:
                image_tensor = image_tensor.squeeze(0)
            image_tensor = image_tensor.unsqueeze(0)

            data_item[self.modality] = image_tensor
            if has_label:
                data_item[label_name] = sample["label"].long()
            return data_item

        if has_label:
            image_slices = []
            label_slices = []
            for slice_id in range(self.n_slices):
                sample = {"image": image[slice_id], "label": label[slice_id]}
                sample = self.transform(sample)
                image_slices.append(sample["image"].float().squeeze(0))
                label_slices.append(sample["label"].long())

            data_item[self.modality] = torch.stack(image_slices, dim=0)
            data_item[label_name] = torch.stack(label_slices, dim=0)
            return data_item

        image_slices = []
        for slice_id in range(self.n_slices):
            sample = self.transform({"image": image[slice_id]})
            image_slices.append(sample["image"].float().squeeze(0))

        data_item[self.modality] = torch.stack(image_slices, dim=0)
        return data_item

    def __len__(self):
        return self.length

    def __del__(self):
        if getattr(self, "data_file", None) is not None:
            self.data_file.close()


class PICAITrainDataset(Dataset):
    def __init__(self, data_root: str, modality: str, n_slices: int):
        self.modality = modality.lower()
        if self.modality not in {"t2w", "adc"}:
            raise ValueError("modality must be 't2w' or 'adc'")

        if n_slices <= 0:
            raise ValueError("n_slices must be a positive integer")

        self.n_slices = n_slices
        self.data_path = os.path.join(data_root, f"unpaired_{self.modality}.h5")
        self.data_file = None

        with h5py.File(self.data_path, "r") as data_file:
            self.patient_images = data_file[self.modality].shape[0]
            self.patient_slices = data_file[self.modality].shape[1]

        if self.n_slices > self.patient_slices:
            raise ValueError("n_slices must not exceed the number of slices per patient")

        self.slice_nums = self.patient_slices - self.n_slices + 1
        self.length = self.patient_images * self.slice_nums

    def _get_data_file(self):
        if self.data_file is None:
            self.data_file = h5py.File(self.data_path, "r")
        return self.data_file

    def __getitem__(self, index: int):
        data_file = self._get_data_file()
        image_index = index // self.slice_nums
        slice_index = index % self.slice_nums

        image = data_file[self.modality][image_index, slice_index:slice_index + self.n_slices]
        data_item = {self.modality: image}

        label_name = "seg"
        if label_name in data_file:
            label = data_file[label_name][image_index, slice_index:slice_index + self.n_slices]
            data_item[label_name] = label.astype(np.int64)

        return data_item

    def __len__(self):
        return self.length

    def __del__(self):
        if getattr(self, "data_file", None) is not None:
            self.data_file.close()


class PICAITestDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        source_modality: str,
        target_modality: str,
        image_transform=None,
    ):
        self.source_modality = source_modality.lower()
        self.target_modality = target_modality.lower()
        self.image_transform = image_transform
        self.data_file = None

        candidate_names = [
            f"paired_{self.source_modality}_{self.target_modality}.h5",
            f"paired_{self.target_modality}_{self.source_modality}.h5",
        ]

        self.data_path = None
        for candidate_name in candidate_names:
            candidate_path = os.path.join(data_root, candidate_name)
            if os.path.exists(candidate_path):
                self.data_path = candidate_path
                break

        if self.data_path is None:
            raise FileNotFoundError(
                f"Could not find paired dataset file under {data_root}. Tried: {candidate_names}"
            )

        with h5py.File(self.data_path, "r") as data_file:
            self.image_nums = data_file[self.source_modality].shape[0]
            self.slice_nums = data_file[self.source_modality].shape[1]

        self.length = self.image_nums

    def _get_data_file(self):
        if self.data_file is None:
            self.data_file = h5py.File(self.data_path, "r")
        return self.data_file

    def __getitem__(self, index: int):
        data_file = self._get_data_file()
        source_image = data_file[self.source_modality][index]
        target_image = data_file[self.target_modality][index]

        if self.image_transform is not None:
            source_image = self.image_transform(source_image)
            target_image = self.image_transform(target_image)

        data_item = {
            self.source_modality: source_image,
            self.target_modality: target_image,
        }

        label_name = "seg"
        if label_name in data_file:
            label = data_file[label_name][index]
            data_item[label_name] = label.astype(np.int64)

        return data_item

    def __len__(self):
        return self.length

    def __del__(self):
        if getattr(self, "data_file", None) is not None:
            self.data_file.close()


PICAITrainDatasetWithTransform = PICAITrainDataset_with_transform
