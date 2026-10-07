# RTMDet 与 FCOS 实现路线

本文记录 YOLO 系列之后的两种水平框检测器，以及一个独立的旋转框研究方向。这里是实施计划和上游代码审阅记录；在 X-DeT 代码、训练和验证完成前，不代表项目已经支持这些模型。

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
- 主要参考配置：[RTMDet-S COCO 配置](https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py)；核心实现位于上游 `mmdet/models/detectors/rtmdet.py` 与 `mmdet/models/dense_heads/rtmdet_head.py`。

### 方法要点

RTMDet 是实时单阶段检测器。参考配置使用 CSPNeXt 主干、CSPNeXt PAFPN，以及在 stride 8/16/32 特征层上工作的无预设宽高 anchor 检测头。检测头通过点位置和到四边的距离解码水平框；训练端以 Dynamic Soft Label Assigner 动态分配正样本，以 Quality Focal Loss 监督分类质量、GIoU Loss 回归框。参考训练配方使用 Cached Mosaic/MixUp、EMA，并在训练后段切换到较弱的增强流水线。

计划实现时保留上述组合的语义，特别是动态软标签分配、回归距离解码、EMA 和训练后段 pipeline switch。先实现水平框版本；不把 RTMDet-R2 的旋转角度、旋转 NMS 或遥感设置混入这个模型。

## FCOS

### 来源

- 论文：[FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)，ICCV 2019。
- 上游实现：[OpenMMLab MMDetection](https://github.com/open-mmlab/mmdetection)。本地参考版本与上述 RTMDet 相同：提交 `cfd5d3a985b0249de009b67d04f37263e11cdf3d`，Apache-2.0。
- 主要参考配置：[FCOS ResNet-50 + FPN COCO 配置](https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py)；核心实现位于上游 `mmdet/models/detectors/fcos.py` 与 `mmdet/models/dense_heads/fcos_head.py`。

### 方法要点

FCOS 不生成 anchor 或候选框，而是在 FPN 的多个空间位置预测类别、LTRB 四边距离和 centerness。不同金字塔层负责不同回归范围；训练时依照点是否位于真值框内及尺度范围分配监督。参考配置使用五个 FPN 层（stride 8、16、32、64、128）、Focal Loss、IoU Loss 和 centerness 二元交叉熵，推理时结合分类分数与 centerness 并执行 NMS。

后续实现需把点坐标、回归范围、centerness 目标、距离到 `xyxy` 的转换和图像缩放还原作为显式教学路径。配置中有不同 FCOS 变体（例如 center sampling、归一化回归距离和 GIoU），第一版将锁定一个配置并记录其与论文及原始 FCOS 实现的差异，不混合不同变体的技巧。

## RTMDet-R2（后续旋转框方向）

- 论文：[RTMDet-R2: An Improved Real-Time Rotated Object Detector](https://github.com/Zeba-Xie/RTMDet-R2) README 所列 PRCV 2023 论文信息。
- 参考仓库：[Zeba-Xie/RTMDet-R2](https://github.com/Zeba-Xie/RTMDet-R2)，本地提交 `20cc7acad5fd98ef4c344c829bb9c360798d8a86`，Apache-2.0；基于 MMRotate/MMDetection 的旋转框扩展。
- 本地 DOTA 配置使用 `RotatedRTMDetSepBNHead`、距离加角度编码、Rotated IoU Loss 与带旋转 IoU 计算器的动态软标签分配；论文报告针对遥感旋转目标。
- 它依赖旋转框数据、角度约定、旋转 IoU/NMS 和相应评估协议，故不作为水平框 RTMDet 的同一模型变体，也不进入当前首轮 VOC 对比。

## 实施顺序与完成条件

1. YOLOv3 当前训练、VOC test 评估和报告完成后，先实现 RTMDet 水平框版，再实现 FCOS。
2. 每个模型提供独立模型、损失/目标分配与后处理代码、recipe、训练/评估/推理入口和版本说明；权重不提交到 Git。
3. 完成 CPU 小样例检查、前向形状与梯度检查、训练恢复检查，再进行明确记录的 VOC 训练和 test 评估。
4. 保存每 10 epoch 检查点、最佳权重、训练曲线、数据分析和预测可视化；报告明确列出环境、命令、权重来源、指标和实现差异。
5. 水平框两种方法完成后，依据目标数据和用户后续优先级单独决定是否开展 RTMDet-R2 的旋转框实现。

目前仅完成上游代码与配置审阅，X-DeT 侧 RTMDet、FCOS 和 RTMDet-R2 均尚未实现或训练。
