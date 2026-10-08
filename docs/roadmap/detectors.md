# RTMDet 与 FCOS 实现及实验路线

本文记录 YOLO 系列之后的两种水平框检测器，以及一个独立的旋转框研究方向。RTMDet-Tiny 和 FCOS 的独立模型/损失/解码实现已落入仓库；完整 VOC 训练、测试和实验结论仍以对应报告为准。

## 共用实验边界

- 以当前 VOC 2007+2012 train/val 合并训练集和 VOC 2007 test 评估集作为首轮共同数据来源，保持类别映射与数据划分一致。
- 每个检测器单独实现和训练。保留论文或参考配置中具有辨识度的网络、标签分配、损失和增强策略，不将其改写成 YOLO 配方。
- 记录实际采用的 ImageNet/检测预训练权重、输入尺寸、优化器、学习率计划、增强、epoch、随机种子和硬件；权重来源及 SHA256 随实验报告保存。
- 统一报告 VOC AP50、AP50:95、每类 AP、训练与验证曲线、数据分析和定性预测图。不同方法的训练配方仍可能不同，因此将同时注明配方差异，不把结果描述为只由模型结构造成。
- 每次先完成一个方法的最小完整路径，再开始另一个方法；训练期间记录资源和 checkpoint，避免两个长任务同时争用 GPU。

## RTMDet（水平框）

### 来源

- 论文：[RTMDet: An Empirical Study of Designing Real-Time Object Detectors](https://arxiv.org/abs/2212.07784)，2022。
- 上游实现：[OpenMMLab MMDetection](https://github.com/open-mmlab/mmdetection)。本地参考版本为 `v3.3.0-5-gcfd5d3a9`，提交 `cfd5d3a985b0249de009b67d04f37263e11cdf3d`，Apache-2.0。
- 主要参考配置：[RTMDet-Tiny COCO 配置](https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py)；核心实现位于上游 `mmdet/models/detectors/rtmdet.py` 与 `mmdet/models/dense_heads/rtmdet_head.py`。

### 方法要点

RTMDet 是实时单阶段检测器。参考配置使用 CSPNeXt 主干、CSPNeXt PAFPN，以及在 stride 8/16/32 特征层上工作的无预设宽高 anchor 检测头。检测头通过点位置和到四边的距离解码水平框；训练端以 Dynamic Soft Label Assigner 动态分配正样本，以 Quality Focal Loss 监督分类质量、GIoU Loss 回归框。参考训练配方使用 Cached Mosaic/MixUp、EMA，并在训练后段切换到较弱的增强流水线。

X-DeT 保留 CSPNeXt、PAFPN、分层 BN 头、动态软标签分配、回归距离解码、QFL 和 GIoU。当前轻量 VOC recipe 尚未实现缓存 Mosaic/MixUp、EMA 和训练后段 pipeline switch，具体差异记录在版本说明；不把 RTMDet-R2 的旋转角度、旋转 NMS 或遥感设置混入这个模型。

## FCOS

### 来源

- 论文：[FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)，ICCV 2019。
- 上游实现：[OpenMMLab MMDetection](https://github.com/open-mmlab/mmdetection)。本地参考版本与上述 RTMDet 相同：提交 `cfd5d3a985b0249de009b67d04f37263e11cdf3d`，Apache-2.0。
- 主要参考配置：[FCOS ResNet-50 + FPN COCO 配置](https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py)；核心实现位于上游 `mmdet/models/detectors/fcos.py` 与 `mmdet/models/dense_heads/fcos_head.py`。

### 方法要点

FCOS 不生成 anchor 或候选框，而是在 FPN 的多个空间位置预测类别、LTRB 四边距离和 centerness。不同金字塔层负责不同回归范围；训练时依照点是否位于真值框内及尺度范围分配监督。参考配置使用五个 FPN 层（stride 8、16、32、64、128）、Focal Loss、IoU Loss 和 centerness 二元交叉熵，推理时结合分类分数与 centerness 并执行 NMS。

X-DeT 显式实现点坐标、回归范围、centerness 目标、距离到 `xyxy` 的转换和图像缩放还原。当前第一版没有启用 FCOSv2 center sampling，回归采用 IoU loss。配置中有不同 FCOS 变体，后续保持变体间差异明确。

## RTMDet-R2（后续旋转框方向）

- 论文：[RTMDet-R2: An Improved Real-Time Rotated Object Detector](https://github.com/Zeba-Xie/RTMDet-R2) README 所列 PRCV 2023 论文信息。
- 参考仓库：[Zeba-Xie/RTMDet-R2](https://github.com/Zeba-Xie/RTMDet-R2)，本地提交 `20cc7acad5fd98ef4c344c829bb9c360798d8a86`，Apache-2.0；基于 MMRotate/MMDetection 的旋转框扩展。
- 本地 DOTA 配置使用 `RotatedRTMDetSepBNHead`、距离加角度编码、Rotated IoU Loss 与带旋转 IoU 计算器的动态软标签分配；论文报告针对遥感旋转目标。
- 它依赖旋转框数据、角度约定、旋转 IoU/NMS 和相应评估协议，故不作为水平框 RTMDet 的同一模型变体，也不进入当前首轮 VOC 对比。

## 下一批：CornerNet 与 CenterNet

用户已指定参考本地 `3dparty/mmdetection/configs/cornernet/` 和 `configs/centernet/`，排在当前 FCOS 完整训练、VOC07 test 评估之后。

- **CornerNet**：以 [CornerNet: Detecting Objects as Paired Keypoints](https://arxiv.org/abs/1808.01244) 为论文基线，对照 `cornernet_hourglass104_8xb6-210e-mstest_coco.py` 和其 COCO checkpoint。独立 Hourglass-104、左上/右下角点热图、corner pooling、associative embedding、offset、配对解码与 Gaussian Soft-NMS 已加入；公开 COCO checkpoint 的匹配比例为 99.999997%，CPU 单步 smoke 已完成，单卡显存/吞吐 smoke 尚待 GPU 空闲后确认。上游模型卡记录 COCO AP 41.2、8×V100、batch 48、训练显存约 15.9 GB，不能直接推定 16 GB 消费卡上的同配方可跑。
- **CenterNet（Objects as Points）**：以 [Objects as Points](https://arxiv.org/abs/1904.07850) 的 `centernet_r18_8xb16-crop512-140e_coco.py` 为基线。ResNet-18、CTResNetNeck（DCNv2 关闭）、中心热图、宽高/offset 标签、loss 和局部峰值解码已加入；公开 COCO checkpoint 的匹配比例为 99.99996%，CPU 单步 smoke 已完成，GPU smoke 尚待 FCOS 结束后确认。
- `configs/centernet/` 还包含 **CenterNet-Update**（R18/R50 + FPN、多尺度检测头、GIoU、1024 LSJ）配置。这是另一种工程/方法变体，不与 `Objects as Points` 基线混称；若纳入实现，单独命名、记录配置与对比。
- 两个方法将沿用 VOC07+12 train/val train、VOC07 val 选权重、VOC07 test 最终评估；优先迁移公开 COCO 检测权重，按类名映射类别头。源码运行不依赖 MMDetection；上游 Apache-2.0 来源、checkpoint 链接与 SHA256 写入版本说明。

## 实施顺序与当前状态

1. YOLOv3 已完成 VOC 训练、test 评估和报告。
2. RTMDet-Tiny 已完成 100 epoch VOC 训练、test 评估、曲线、数据分析和预测可视化；结果见 [`RTMDet-Tiny VOC 报告`](../reports/rtmdet_voc0712_tiny_20261008.md)。
3. RTMDet 与 FCOS 均已有独立模型、损失/目标分配、后处理、recipe 和训练/评估入口；CPU 小样例检查了前向形状、损失/梯度和解码。
4. FCOS ResNet-50 目前按 120 epoch 配置训练中。完成后需用 validation best 在 VOC 2007 test 上评估，并生成曲线、数据分析、预测可视化与实验报告；checkpoint 和本地实验产物不纳入源码提交。
5. FCOS 完成后，先匹配公开 COCO checkpoint，再依次做 CenterNet 与 CornerNet 的 GPU smoke、显存/吞吐测量和训练；按 validation best 在 VOC07 test 最终评估，生成曲线、预测图、数据分析和报告。
6. CornerNet 与 CenterNet 首轮完成后，再依据目标数据及实验结果决定是否开展 RTMDet-R2 旋转框和 CenterNet-Update 变体。
