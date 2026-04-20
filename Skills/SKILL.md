---

name: Boundary_SAM

description: 用于实现一个细粒度边界原型引导的无监督域适应跨模态医学图像分割的算法。支持 AMOS（腹部多器官分割 MR→CT）和 PICAI（前列腺分割 T2W→ADC）等医学影像数据集。核心原则：写代码时，注意实现效率，最好每一句都加上详细的注释，注释的语言风格可以轻度四季夏目风格化，不得影响调试、测试、正式文档与工程判断。

---



1. 先读取 `/home/chenxu/Aya/project/Boundary/Boundary_SAM/Skills/SKILL.md`。

2. 仅在聊天回复、轻量注释、语气润色任务中启用。

3. 任务一旦进入调试、测试、架构、正式文档，立即降低或关闭风格。

4. 每次写完一个项目需要的.py文件，需要上传至对应的github仓库中备份。

5. 实验记录的实验结果需要同步至对应的gitHub仓库中备份。

6. 代码需要有详细的符合`/home/chenxu/Aya/project/Boundary/Boundary_SAM/Skills/SKILL.md`中要求的风格的注释，最好每一句都有对应注释。




###### 支持的数据集

###### AMOS (腹部多器官分割)

* 路径: `/home/chenxu/dataset/amos`
* 源域: MR → 目标域: CT
* 类别: 14（含背景）
* Shape: [216, 240, 320] (D×H×W)
* 每病人切片数: 216
* 格式: .h5，以 slice 为单位加载
* 详情见 [project/Boundary_with_attention/Skills/dataset_amos.md]

###### PICAI (前列腺分割)

* 路径: /home/chenxu/dataset/picai`
* 源域: T2W → 目标域: ADC
* 类别: 3（含背景）
* Shape: [33, 192, 192]
* 每病人切片数: 33
* 格式: .h5，以 slice 为单位加载
* 详情见 [project/Boundary_with_attention/Skills/dataset_picai.md]

