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
    # 先随机转 0/90/180/270 度，这种增强对医学切片通常比较安全。
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    # 再随机沿一个空间轴翻转一下，让模型别太依赖固定方向。
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()

    if label is None:
        return image

    # 标签要跟图像做完全相同的几何变换，不然监督就散了。
    label = np.rot90(label, k)
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image: np.ndarray, label: Optional[np.ndarray] = None, angle_range: Tuple[int, int] = (-20, 20)):
    # 角度范围不设太大，轻轻转一下就够，太狠了反而像在故意刁难数据。
    angle = np.random.randint(angle_range[0], angle_range[1] + 1)
    # 图像插值用 order=1，平滑一点，视觉上会自然些。
    image = ndimage.rotate(image, angle, order=1, reshape=False, mode="nearest")

    if label is None:
        return image

    # 标签必须坚持最近邻插值，不然类别边界会被抹得一塌糊涂。
    label = ndimage.rotate(label, angle, order=0, reshape=False, mode="nearest")
    return image, label


def gaussian_noise(image: np.ndarray, label: Optional[np.ndarray] = None, std: float = 0.05):
    # 高斯噪声模拟扫描扰动，幅度控制得比较克制。
    noise = np.random.normal(0.0, std, image.shape)
    image = image + noise
    if label is None:
        return image
    return image, label


def gaussian_blur(image: np.ndarray, label: Optional[np.ndarray] = None, sigma_range: Tuple[float, float] = (0.0, 1.0)):
    # 模糊增强模拟成像清晰度波动，sigma 随机采样。
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    image = gaussian_filter(image, sigma=sigma, order=0)
    if label is None:
        return image
    return image, label


def gamma_correction(image: np.ndarray, label: Optional[np.ndarray] = None, gamma_range: Tuple[float, float] = (0.7, 1.5)):
    # gamma 主要在强度分布上做文章，让模型别太迷信固定对比度。
    gamma_value = random.random() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
    image_min = image.min()
    image_max = image.max()

    # 只有确实存在动态范围时才做归一化，不然会出现除零这种无聊事故。
    if image_min < image_max:
        image = (image - image_min) / (image_max - image_min)
        # 偶尔做一下反相，给模型一点小惊喜，但不至于太离谱。
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
    # 对比度增强围绕均值拉伸或压缩，幅度故意控制在温和区间。
    factor = np.random.uniform(contrast_range[0], contrast_range[1])
    mean = image.mean()
    image_min = image.min()
    image_max = image.max()
    image = (image - mean) * factor + mean
    # 最后把值裁回原范围，免得越增强越失控。
    image = np.clip(image, image_min, image_max)

    if label is None:
        return image
    return image, label


def bernstein_poly(i: int, n: int, t: np.ndarray):
    # Bezier 曲线的基础 Bernstein 多项式，非线性强度变换要靠它搭桥。
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points: Sequence[Sequence[float]], n_times: int = 1000):
    # 给一组控制点，采样出平滑曲线，后面拿来做 intensity remap。
    x_points = np.array([point[0] for point in points])
    y_points = np.array([point[1] for point in points])
    t = np.linspace(0.0, 1.0, n_times)
    polynomial_array = np.array([bernstein_poly(i, len(points) - 1, t) for i in range(len(points))])
    x_values = np.dot(x_points, polynomial_array)
    y_values = np.dot(y_points, polynomial_array)
    return x_values, y_values


def nonlinear_transformation(image: np.ndarray):
    # 随机控制点生成一条单调或近单调的映射曲线，用来做非线性灰度变换。
    points = [[0.0, 0.0], [random.random(), random.random()], [random.random(), random.random()], [1.0, 1.0]]
    x_values, y_values = bezier_curve(points, n_times=1000)

    # 一半概率只排序 x，一半概率 x/y 都排序，等于给变换多一点变化空间。
    if random.random() < 0.5:
        x_values = np.sort(x_values)
    else:
        x_values = np.sort(x_values)
        y_values = np.sort(y_values)

    # 最终按曲线做插值映射，把原图强度重新折腾一遍。
    return np.interp(image, x_values, y_values)


class RandomGenerator:
    """
    `augmentation_mode` controls how much augmentation is used:
    - `none`: only resize + tensor conversion
    - `basic`: spatial augmentation
    - `fine`: spatial + intensity + nonlinear augmentation
    """

    def __init__(self, output_size: Optional[Sequence[int]] = (192, 192), augmentation_mode: str = "none"):
        # output_size 决定最终切片尺寸；如果传 None，就尊重原始大小。
        self.output_size = tuple(output_size) if output_size is not None else None
        # augmentation_mode 控制增强强度，后面训练脚本只改这个参数就够。
        self.augmentation_mode = augmentation_mode.lower()
        if self.augmentation_mode not in {"none", "basic", "fine"}:
            raise ValueError("augmentation_mode must be one of {'none', 'basic', 'fine'}")

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        # 没指定输出尺寸就不 resize，别多此一举。
        if self.output_size is None:
            return image
        if image.shape == tuple(self.output_size):
            return image

        # 图像 resize 用线性插值，视觉更自然一些。
        scale = (self.output_size[0] / image.shape[0], self.output_size[1] / image.shape[1])
        return zoom(image, scale, order=1)

    def _resize_label(self, label: np.ndarray) -> np.ndarray:
        # 标签同理，如果尺寸本来就对，就别碰它。
        if self.output_size is None:
            return label
        if label.shape == tuple(self.output_size):
            return label

        # 标签 resize 必须用最近邻，这个规矩还是要守。
        scale = (self.output_size[0] / label.shape[0], self.output_size[1] / label.shape[1])
        return zoom(label, scale, order=0)

    def __call__(self, sample: Dict[str, np.ndarray]):
        # sample 至少包含 image；训练时一般还会带 label。
        image = sample["image"]
        label = sample.get("label")

        # basic/fine 都包含几何增强，这部分风险低、收益稳定。
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

        # fine 模式额外加上强度域扰动，适合你说的“精细增强”场景。
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

        # 最后统一 resize 并转成 float32，免得 DataLoader 后面再闹 dtype 脾气。
        image = self._resize_image(image).astype(np.float32)
        image_tensor = torch.from_numpy(image).unsqueeze(0)

        output = {"image": image_tensor}
        if label is not None:
            # 标签统一转 int64，后面交给 CrossEntropyLoss 最省心。
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
        # 模态只允许 t2w / adc，两者之外都不是这个数据集该处理的东西。
        self.modality = modality.lower()
        if self.modality not in {"t2w", "adc"}:
            raise ValueError("modality must be 't2w' or 'adc'")

        # n_slices 代表一次取连续多少张切片，必须是正数。
        if n_slices <= 0:
            raise ValueError("n_slices must be a positive integer")

        self.n_slices = n_slices
        self.output_size = tuple(output_size) if output_size is not None else None
        # 如果外面没传自定义 transform，就按 augmentation_mode 自动建一个。
        self.transform = transform if transform is not None else RandomGenerator(output_size, augmentation_mode)
        self.data_path = os.path.join(data_root, f"unpaired_{self.modality}.h5")
        # h5 文件句柄延迟到真正取样时再打开，DataLoader 多进程时会更稳一点。
        self.data_file = None

        # 这里只读取元信息，不提前把整个数据文件抱进内存。
        with h5py.File(self.data_path, "r") as data_file:
            self.patient_images = data_file[self.modality].shape[0]
            self.patient_slices = data_file[self.modality].shape[1]

        # 连续切片数不能比单个病人的切片总数还大，这个不讲道理的输入直接拦掉。
        if self.n_slices > self.patient_slices:
            raise ValueError("n_slices must not exceed the number of slices per patient")

        # 每个病人可以滑出的窗口数量。
        self.slice_nums = self.patient_slices - self.n_slices + 1
        # 数据集总长度 = 病人数 * 每个病人的滑窗数。
        self.length = self.patient_images * self.slice_nums

    def _get_data_file(self):
        # 第一次访问样本时再打开文件句柄，避免初始化阶段过早占资源。
        if self.data_file is None:
            self.data_file = h5py.File(self.data_path, "r")
        return self.data_file

    def __getitem__(self, index: int):
        data_file = self._get_data_file()
        # 先把全局 index 拆成“第几个病人 + 从哪张切片开始”。
        image_index = index // self.slice_nums
        slice_index = index % self.slice_nums

        # 取出连续 n_slices 张切片，形状通常是 [n_slices,H,W]。
        image = data_file[self.modality][image_index, slice_index:slice_index + self.n_slices]
        data_item = {}

        label_name = "seg"
        has_label = label_name in data_file
        label = None
        if has_label:
            # 训练集如果带标签，就一起取出来；标签转 int64 方便后面做 CE。
            label = data_file[label_name][image_index, slice_index:slice_index + self.n_slices].astype(np.int64)

        # 单切片模式最常见，直接按 2D 样本走 transform 即可。
        if self.n_slices == 1:
            sample = {"image": image[0]}
            if has_label:
                sample["label"] = label[0]
            sample = self.transform(sample)

            image_tensor = sample["image"].float()
            # transform 输出是 [1,H,W]，这里整理成数据集约定的 [1,H,W]。
            if image_tensor.ndim == 3 and image_tensor.shape[0] == 1:
                image_tensor = image_tensor.squeeze(0)
            image_tensor = image_tensor.unsqueeze(0)

            data_item[self.modality] = image_tensor
            if has_label:
                data_item[label_name] = sample["label"].long()
            return data_item

        # 多切片模式下，逐 slice 做 2D 增强，再堆回去，逻辑最稳，不容易把维度玩坏。
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

        # 没标签时只处理图像分支，通常用于目标域无监督数据。
        image_slices = []
        for slice_id in range(self.n_slices):
            sample = self.transform({"image": image[slice_id]})
            image_slices.append(sample["image"].float().squeeze(0))

        data_item[self.modality] = torch.stack(image_slices, dim=0)
        return data_item

    def __len__(self):
        return self.length

    def __del__(self):
        # 数据集销毁时顺手把 h5 句柄关掉，安静一点，省得资源挂着不走。
        if getattr(self, "data_file", None) is not None:
            self.data_file.close()


class PICAITrainDataset(Dataset):
    def __init__(self, data_root: str, modality: str, n_slices: int):
        # 这是不带 transform 的朴素版本，适合你想自己在外面接增强的时候用。
        self.modality = modality.lower()
        if self.modality not in {"t2w", "adc"}:
            raise ValueError("modality must be 't2w' or 'adc'")

        if n_slices <= 0:
            raise ValueError("n_slices must be a positive integer")

        self.n_slices = n_slices
        self.data_path = os.path.join(data_root, f"unpaired_{self.modality}.h5")
        self.data_file = None

        # 只读元信息，避免初始化时做多余 IO。
        with h5py.File(self.data_path, "r") as data_file:
            self.patient_images = data_file[self.modality].shape[0]
            self.patient_slices = data_file[self.modality].shape[1]

        if self.n_slices > self.patient_slices:
            raise ValueError("n_slices must not exceed the number of slices per patient")

        self.slice_nums = self.patient_slices - self.n_slices + 1
        self.length = self.patient_images * self.slice_nums

    def _get_data_file(self):
        # 延迟打开 h5，跟上面那个带 transform 的版本保持一致。
        if self.data_file is None:
            self.data_file = h5py.File(self.data_path, "r")
        return self.data_file

    def __getitem__(self, index: int):
        data_file = self._get_data_file()
        # 一样先拆 index，定位到病人和切片起点。
        image_index = index // self.slice_nums
        slice_index = index % self.slice_nums

        # 这里直接返回 numpy 切片，不做额外加工。
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
        # 句柄该关还是要关，懒是可以，资源泄漏不行。
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
        # 测试/验证集是 paired 数据，所以这里同时关心 source 和 target 两个模态。
        self.source_modality = source_modality.lower()
        self.target_modality = target_modality.lower()
        self.image_transform = image_transform
        self.data_file = None

        # 数据文件名在旧工程里存在顺序不统一的问题，所以两种命名都试一下。
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

        # 两个名字都找不到，那就明确报错，别让后面静悄悄地出事故。
        if self.data_path is None:
            raise FileNotFoundError(
                f"Could not find paired dataset file under {data_root}. Tried: {candidate_names}"
            )

        # 这里只取病人数和切片数，用来定义 __len__ 和后续迭代逻辑。
        with h5py.File(self.data_path, "r") as data_file:
            self.image_nums = data_file[self.source_modality].shape[0]
            self.slice_nums = data_file[self.source_modality].shape[1]

        self.length = self.image_nums

    def _get_data_file(self):
        # 还是延迟打开，毕竟 h5 句柄不是什么值得囤的收藏品。
        if self.data_file is None:
            self.data_file = h5py.File(self.data_path, "r")
        return self.data_file

    def __getitem__(self, index: int):
        data_file = self._get_data_file()
        # paired 数据一次返回同一个病人的 source / target volume。
        source_image = data_file[self.source_modality][index]
        target_image = data_file[self.target_modality][index]

        if self.image_transform is not None:
            # 如果外面传了图像变换，就分别作用到两个模态上。
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
        # 结束时关文件句柄，干净一点，对谁都好。
        if getattr(self, "data_file", None) is not None:
            self.data_file.close()


# 给一个兼容旧命名的别名，后面迁移训练代码时能省不少麻烦。
PICAITrainDatasetWithTransform = PICAITrainDataset_with_transform
