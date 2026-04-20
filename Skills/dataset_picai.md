# PICAI 数据集说明

## 基本信息

| 属性 | 值 |
|------|-----|
| 路径 | `/home/chenxu/dataset/picai` |
| 任务 | 前列腺分割 (Prostate Segmentation) |
| 源域 | T2W (T2加权像) |
| 目标域 | ADC (表观扩散系数) |
| 类别数 | 3 (含背景) |
| 数据格式 | .h5 (HDF5) |
| 加载方式 | 以切片 (slice) 为单位加载 |
| **每病人切片数** | **33** |

## 数据尺寸

- 单病人数据: `[33, 192, 192]`
- 每个病人有 **33 个切片**
- 每个切片: `192 × 192`

## H5 文件 Key

- `'t2w'`: T2W 图像数据
- `'adc'`: ADC 图像数据
- `'seg'`: 分割标签

## 预处理状态

数据已处理完毕，**不需要额外归一化操作**。

## 目录结构

```
picai/
├── train/
│   ├── unpaired_t2w.h5     # 源域 T2W 训练数据 (unpaired)
│   └── unpaired_adc.h5     # 目标域 ADC 训练数据 (unpaired)
├── valid/
│   └── paired_t2w_adc.h5   # 成对验证数据
└── test/
    └── paired_t2w_adc.h5   # 成对测试数据
```

## 使用注意

- 训练集为 unpaired (非配对) 数据
- 验证集和测试集为 paired (成对) 数据
- 验证/测试时需以**单个病人的 33 个切片合并为 volume** 进行 3D 验证
- 加载时以单个切片为单位，但验证/测试时需还原为 volume
