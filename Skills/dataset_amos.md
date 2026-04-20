# AMOS 数据集说明

## 基本信息

| 属性 | 值 |
|------|-----|
| 路径 | `/home/chenxu/dataset/amos` |
| 任务 | 腹部多器官分割 (Multi-organ Segmentation) |
| 源域 | MR (磁共振) |
| 目标域 | CT (计算机断层扫描) |
| 配对方式 | Unpaired (非配对) |
| 类别数 | 14 (含背景) |
| 数据格式 | .h5 (HDF5) |
| 数据粒度 | 以 slice 为单位加载 |
| **每病人切片数** | **216** |

## 数据尺寸

- 单个 volume 尺寸: `[216, 240, 320]` 对应 `(Depth × Height × Width)`
- 每个切片: `240 × 320`
- 每个病人: **216 个切片**

## 预处理状态

数据已完成以下预处理，**可直接用于模型训练**：
- 强度归一化 (Normalization)
- 尺寸对齐与裁剪 (统一为固定体素大小)
- 标签格式标准化

## 目录结构

```
amos/
├── train/
│   ├── unpaired_mr.h5      # 源域 MR 训练数据
│   └── unpaired_ct.h5      # 目标域 CT 训练数据
├── valid/
│   ├── unpaired_mr.h5      # 源域 MR 验证数据
│   └── unpaired_ct.h5      # 目标域 CT 验证数据
└── test/
    └── unpaired_ct.h5      # 目标域 CT 测试数据
```

## H5 文件结构

每个 .h5 文件包含多个病人的 3D volume。

**Keys:**
- `'ct'` / `'mr'`: 图像数据
- `'ct_seg'` / `'mr_seg'`: 分割标签

## 使用注意

- MR 与 CT 数据来自不同病人，不一一对应
- 测试集仅包含 CT 数据（用于目标域评估）
- 验证/测试时应以单个病人的 **216 个切片合并为 volume** 进行 3D 验证
