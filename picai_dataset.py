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
    rotate_k = np.random.randint(0, 4)  # 随机决定旋转 0/90/180/270 度。
    flip_axis = np.random.randint(0, 2)  # 随机决定沿哪一个轴翻转。
    image = np.rot90(image, rotate_k)  # 先做离散旋转，这一步不会引入插值误差。
    image = np.flip(image, axis=flip_axis).copy()  # 再做翻转，并 copy 保证内存连续。
    if label is None:  # 没有标签时只返回图像，别多想。
        return image, None
    label = np.rot90(label, rotate_k)  # 标签必须做同样的几何变换，不然就对不上。
    label = np.flip(label, axis=flip_axis).copy()  # 同样 copy 一次，后面转 tensor 更稳。
    return image, label


def random_rotate(image: np.ndarray, label: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """随机小角度旋转，增强视角扰动。"""
    angle = np.random.randint(-20, 21)  # 角度范围和旧实现保持同一级别，别拧太狠。
    image = ndimage.rotate(image, angle, order=1, reshape=False)  # 图像用线性插值，边缘过渡会自然一点。
    if label is None:  # 没有标签时直接结束，这很正常。
        return image, None
    label = ndimage.rotate(label, angle, order=0, reshape=False)  # 标签必须用最近邻插值，不然类别会污染。
    return image, label


def gamma_correction(image: np.ndarray) -> np.ndarray:
    """随机 gamma 变换，轻度扰动强度分布。"""
    image_min = float(image.min())  # 先记住最小值，后面要恢复原始范围。
    image_max = float(image.max())  # 同理把最大值也留下。
    if image_max <= image_min:  # 退化图像没有操作意义，别硬改。
        return image
    gamma = float(np.random.uniform(0.7, 1.5))  # 范围和旧代码一致，够用了。
    image = (image - image_min) / (image_max - image_min)  # 先归一化到 0~1，不然 gamma 没法讲道理。
    if np.random.rand() < 0.5:  # 随机反相一下，增加一点强度多样性。
        image = 1.0 - image
    image = np.power(image, gamma)  # 真正的 gamma 变换在这里。
    image = image * (image_max - image_min) + image_min  # 再映射回原来的数值范围。
    return image


def contrast_augment(image: np.ndarray) -> np.ndarray:
    """轻度对比度增强，保持上下界不跑飞。"""
    image_mean = float(image.mean())  # 以均值为中心做缩放，比较稳。
    image_min = float(image.min())  # 记录下界，后面做裁剪。
    image_max = float(image.max())  # 记录上界，后面做裁剪。
    factor = float(np.random.uniform(0.9, 1.1))  # 轻一点就够了，过头反而伤分布。
    image = (image - image_mean) * factor + image_mean  # 围绕均值做线性拉伸。
    image = np.clip(image, image_min, image_max)  # 把结果限制回原范围，省得出现离谱值。
    return image


def gaussian_blur(image: np.ndarray) -> np.ndarray:
    """轻度高斯模糊，模拟采样和重建差异。"""
    sigma = float(np.random.uniform(0.0, 1.0))  # 模糊强度随机一点，但不要太重。
    return gaussian_filter(image, sigma=sigma, order=0)  # 这里只改图像，不碰标签。


def gaussian_noise(image: np.ndarray) -> np.ndarray:
    """加一点高斯噪声，别把网络宠坏。"""
    noise = np.random.normal(loc=0.0, scale=0.05, size=image.shape)  # 方差沿用旧实现的量级。
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
        self.output_size = tuple(output_size)  # 输出大小统一保存成 tuple，后面别再改了。
        self.spatial_aug = SpatialAug  # 保留旧参数名的同时，用更直白的成员变量。
        self.intensity_aug = IntensityAug  # 强度增强开关单独存下来。
        self.nonlinear_aug = NonlinearAug  # 旧接口里有这个开关，这里先兼容住。

    def _resize(self, image: np.ndarray, label: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """把切片缩放到目标大小。"""
        height, width = image.shape  # 当前尺寸先拿出来，后面算缩放比要用。
        target_h, target_w = self.output_size  # 目标尺寸就两维，不需要绕弯。
        image = zoom(image, (target_h / height, target_w / width), order=1)  # 图像用线性插值，更自然一点。
        if label is None:  # 没有标签就只缩图像，不做多余动作。
            return image, None
        label = zoom(label, (target_h / height, target_w / width), order=0)  # 标签继续坚持最近邻插值。
        return image, label

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """执行增强，并把结果转成训练可直接使用的 tensor。"""
        image = sample["image"]  # 输入图像必须存在，这个不用怀疑。
        label = sample.get("label")  # 标签可能不存在，所以这里用 get。

        if self.nonlinear_aug and np.random.rand() > 0.5:  # 先保留开关位，避免接口缺口。
            image = gamma_correction(image)  # 这里用 gamma 代替复杂的非线性映射，简洁一点更可控。

        if self.spatial_aug:  # 空间增强单独控制，排查问题时方便关掉。
            if np.random.rand() > 0.5:  # 旋转翻转是主力增强，命中率高一点无所谓。
                image, label = random_rot_flip(image, label)
            elif np.random.rand() > 0.5:  # 小角度旋转放在次级分支，别每次都转。
                image, label = random_rotate(image, label)

        if self.intensity_aug:  # 强度增强只作用在图像上，标签不参与这些闹腾。
            if np.random.rand() > 0.7:
                image = gamma_correction(image)
            if np.random.rand() > 0.7:
                image = contrast_augment(image)
            if np.random.rand() > 0.7:
                image = gaussian_blur(image)
            if np.random.rand() > 0.5:
                image = gaussian_noise(image)

        image, label = self._resize(image, label)  # 最后再统一缩放，流程更干净。
        image_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # 图像转成 [1, H, W]，和旧训练代码对齐。
        result = {"image": image_tensor}  # 先把图像放进去，标签后补。
        if label is not None:  # 有标签再转 long，交叉熵就喜欢这个类型。
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
        self.data_root = _to_path(data_root)  # 根目录先规范化，路径问题少一点。
        self.modality = modality.lower()  # 模态统一成小写，避免传参时大小写打架。
        self.n_slices = int(n_slices)  # 连续切片数量强制转 int，省得后面算长度出事。
        self.transform = transform  # 增强器可以为空，这很常见。
        if self.modality not in {"t2w", "adc"}:  # PICAI 这里只有这两个模态，别随手乱传。
            raise ValueError(f"Unsupported modality: {modality}")
        if self.n_slices < 1:  # 连一张切片都不取就没意义了。
            raise ValueError("n_slices must be >= 1")

        self.file_path = self.data_root / f"unpaired_{self.modality}.h5"  # 训练文件名是固定模式，直接拼出来。
        self._file: Optional[h5py.File] = None  # h5 文件句柄延迟打开，DataLoader 多进程时更稳。

        with h5py.File(self.file_path, "r") as data_file:  # 这里只短暂打开一次，用来读取元信息。
            self.patient_count = int(data_file[self.modality].shape[0])  # 病人数就是第一维。
            self.slice_count = int(data_file[self.modality].shape[1])  # 每个病人的切片数就是第二维。
            self.has_label = "seg" in data_file  # 训练文件里通常带标签，但还是老老实实检查一下。

        self.window_count = self.slice_count - self.n_slices + 1  # 每个病人可滑出的窗口数量。
        if self.window_count < 1:  # 切片窗口比 volume 还长就不对了。
            raise ValueError("n_slices is larger than the available slice count")
        self.length = self.patient_count * self.window_count  # 总长度 = 病人数 * 每病人窗口数。

    def _get_file(self) -> h5py.File:
        """按需打开 h5 文件。"""
        if self._file is None:  # 第一次访问时再打开，避免 Dataset 被 fork 时句柄混乱。
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def _apply_transform(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """把增强逻辑单独收起来，主流程会清楚很多。"""
        if self.transform is None:  # 不增强时直接转 tensor，别搞复杂。
            image_tensor = torch.from_numpy(image.astype(np.float32))
            label_tensor = None if label is None else torch.from_numpy(label.astype(np.int64))
            return image_tensor, label_tensor

        if label is None:  # 没有标签时也要保持切片窗口长度不变。
            image_list = []  # 每张切片都要独立过一遍增强流程。
            for slice_index in range(self.n_slices):  # 逐张处理，行为才稳定。
                dummy_label = np.zeros_like(image[slice_index], dtype=np.int64)  # 这里的占位标签只是为了复用统一接口。
                transformed = self.transform({"image": image[slice_index], "label": dummy_label})
                image_list.append(transformed["image"].float())  # 每张图像都是 [1, H, W]。
            if self.n_slices == 1:  # 单切片时直接返回 [1, H, W]，继续兼容旧代码。
                return image_list[0], None
            image_tensor = torch.stack(image_list, dim=0).squeeze(1)  # 多切片时堆成 [S, H, W]。
            return image_tensor, None

        if self.n_slices == 1:  # 单切片最简单，直接按 2D 图像处理。
            transformed = self.transform({"image": image[0], "label": label[0]})
            return transformed["image"].float(), transformed["label"].long()

        image_list = []  # 多切片时逐张增强，别试图偷懒共享随机变换。
        label_list = []  # 标签也逐张同步处理，不然空间对应关系会乱。
        for slice_index in range(self.n_slices):  # 遍历这一小包连续切片。
            transformed = self.transform({"image": image[slice_index], "label": label[slice_index]})
            image_list.append(transformed["image"].float())  # 每张图像都是 [1, H, W]。
            label_list.append(transformed["label"].long())  # 每张标签都是 [H, W]。

        image_tensor = torch.stack(image_list, dim=0).squeeze(1)  # 堆成 [S, H, W]，和旧代码保持一致。
        label_tensor = torch.stack(label_list, dim=0)  # 标签自然也是 [S, H, W]。
        return image_tensor, label_tensor

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        data_file = self._get_file()  # 先拿到当前 worker 对应的文件句柄。
        patient_index = index // self.window_count  # 算出落在哪个病人上。
        slice_index = index % self.window_count  # 算出这个病人的窗口起点。

        image = data_file[self.modality][patient_index, slice_index:slice_index + self.n_slices]  # 取连续切片窗口。
        label = None  # 先默认没有标签，后面按需补上。
        if self.has_label:  # 训练文件带标签时就一起读出来。
            label = data_file["seg"][patient_index, slice_index:slice_index + self.n_slices]

        image_tensor, label_tensor = self._apply_transform(image=image, label=label)  # 图像和标签一起处理，别分开乱来。
        item = {self.modality: image_tensor}  # 返回字典键名沿用旧训练脚本，迁移时更省心。
        if label_tensor is not None:  # 有标签才塞进返回值，这样兼容性更自然。
            item["seg"] = label_tensor
        return item

    def __len__(self) -> int:
        return self.length  # 长度已经在初始化时算好了，直接返回就行。

    def __del__(self) -> None:
        if getattr(self, "_file", None) is not None:  # 句柄存在时再关，省得析构时报空指针。
            try:
                self._file.close()  # h5 文件要手动关掉，不然总让人不放心。
            except Exception:
                pass  # 析构阶段别再制造额外异常，没必要。


class PICAIPairedVolumeDataset(Dataset):
    """PICAI 验证/测试集，按整例 volume 读取。"""

    def __init__(
        self,
        data_root: str | Path,
        source_modality: str,
        target_modality: str,
        image_transform=None,
    ) -> None:
        self.data_root = _to_path(data_root)  # 路径先规范化，后续更省事。
        self.source_modality = source_modality.lower()  # 源域名字统一小写。
        self.target_modality = target_modality.lower()  # 目标域名字也统一小写。
        self.image_transform = image_transform  # 这里通常只做 tensor 化，增强一般不在验证里乱用。
        self.file_path = self._resolve_file_path()  # paired 文件名顺序在不同脚本里不完全统一，自动兜底更省心。
        self._file: Optional[h5py.File] = None  # 句柄同样延迟打开。

        with h5py.File(self.file_path, "r") as data_file:  # 先读元信息。
            self.length = int(data_file[self.source_modality].shape[0])  # 每个样本就是一个病人 volume。
            self.has_label = "seg" in data_file  # 验证和测试常常带标签，但还是检查一下。

    def _resolve_file_path(self) -> Path:
        """自动匹配 paired 文件名，别让命名顺序这种小事卡住实验。"""
        candidates = [
            self.data_root / f"paired_{self.target_modality}_{self.source_modality}.h5",  # 这是旧代码实际使用过的顺序。
            self.data_root / f"paired_{self.source_modality}_{self.target_modality}.h5",  # 这是更直觉的顺序，也顺手兼容掉。
        ]
        for candidate in candidates:  # 逐个尝试，找到就结束。
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Cannot find paired PICAI file under {self.data_root}")  # 都找不到时就明确报错，别沉默失败。

    def _get_file(self) -> h5py.File:
        """按需打开 h5 文件。"""
        if self._file is None:  # 首次访问时打开一次就够。
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """把 numpy volume 转成 float tensor。"""
        if self.image_transform is not None:  # 如果外部显式给了变换，就尊重它。
            return self.image_transform(array)
        return torch.from_numpy(array.astype(np.float32))  # 默认直接转 float tensor，简单但够用。

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        data_file = self._get_file()  # 拿文件句柄。
        source_image = self._to_tensor(data_file[self.source_modality][index])  # 读源域整例 volume。
        target_image = self._to_tensor(data_file[self.target_modality][index])  # 读目标域整例 volume。
        item = {self.source_modality: source_image, self.target_modality: target_image}  # 组装返回字典。
        if self.has_label:  # 有标签就一起给出，评估要靠它。
            item["seg"] = torch.from_numpy(data_file["seg"][index].astype(np.int64))
        return item

    def __len__(self) -> int:
        return self.length  # 总样本数就是病人数。

    def __del__(self) -> None:
        if getattr(self, "_file", None) is not None:  # 句柄存在时再关闭。
            try:
                self._file.close()  # 该关还是得关，别把文件系统当空气。
            except Exception:
                pass  # 析构时安静一点。


class PICAITrainDataset_with_transform(PICAIUnpairedSliceDataset):
    """兼容旧训练脚本的类名，接口不变。"""


class PICAITrainDataset(PICAIUnpairedSliceDataset):
    """兼容旧训练脚本的类名，接口不变。"""

    def __init__(self, data_root: str | Path, modality: str, n_slices: int) -> None:
        super().__init__(data_root=data_root, modality=modality, n_slices=n_slices, transform=None)


class PICAITestDataset(PICAIPairedVolumeDataset):
    """兼容旧训练脚本的类名，接口不变。"""


RandomGenerator = RandomSliceTransform  # 旧代码里就是这个名字，保留掉，省得你后面改一串导入。

