# YOLOv3-head + ConvNeXt-Small：VOC 训练与测试报告

## 结果

| 数据集划分 | 图像数 | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: |
| VOC 2007 val | 2,510 | 0.7993 | 0.4476 |
| VOC 2007 test | 4,952 | **0.7937** | **0.4454** |

最终测试使用验证集 mAP@0.5:0.95 最佳的 epoch 185 checkpoint。测试集共 12,032 个标注框，模型输出 85,171 个候选检测（评估时每张图最多保留 100 框）。类别 AP、训练曲线、数据分析和测试集可视化保存在对应运行目录中。

## 实验设置

- 数据：VOC 2007 train + VOC 2012 trainval（14,041 张 / 33,751 框）训练；VOC 2007 val 选 checkpoint；VOC 2007 test 做最终评估。
- 输入：固定 416×416，将图像缩放到正方形；数据增强沿用项目 YOLOv3 配方。
- 模型：项目 YOLOv3 检测头和多尺度 anchor 路径，主干替换为 TorchVision ImageNet1K 预训练 ConvNeXt-Small。**这是 ConvNeXt 主干的 YOLOv3-head 工程变体，不是原论文 Darknet-53 结构的严格复现。**
- 训练：计划 200 epochs，前 10 epochs 冻结主干，之后主干学习率为检测头的 0.1 倍；416 输入、micro batch 16、梯度累积后有效 batch 64、BF16、SGD、初始 LR 0.001，epoch 160 和 180 各衰减 10 倍。每 10 epochs 保存 checkpoint，每 5 epochs 在验证集评估。
- 恢复：最终训练进程从 epoch 140 checkpoint 续训至 epoch 200；运行配置中记载了 resume 路径。
- 运行设备：NVIDIA GeForce RTX 4060 Ti 16GB，PyTorch 2.10.0+cu128，Python 3.10.12。

## 学习率与验证曲线观察

| Epoch | 检测头 LR | 验证 mAP@0.5 | 验证 mAP@0.5:0.95 |
| ---: | ---: | ---: | ---: |
| 155 | 1e-3 | 0.7821 | 0.4006 |
| 160 | 1e-4 | 0.7867 | 0.4093 |
| 165 | 1e-4 | 0.8006 | 0.4450 |
| 175 | 1e-4 | 0.8005 | 0.4452 |
| 180 | 1e-5 | 0.7972 | 0.4430 |
| **185** | **1e-5** | **0.7993** | **0.4476** |
| 200 | 1e-5 | 0.7995 | 0.4454 |

epoch 160 的学习率下降后，mAP@0.5:0.95 在接下来五轮明显上升；epoch 180 再降学习率后，定位指标仍在小范围波动，epoch 185 得到最佳验证值。最终 test 与最佳 val 的 mAP@0.5:0.95 相差约 0.0023，当前证据没有显示明显的验证集过拟合。后续选择权重应使用 `best.pt`，而不是最后一轮的 `last.pt`。

## 与已跑版本的参考

同一 VOC 2007 test 划分上的旧结果为：YOLOv1 ConvNeXt-Small 工程版 mAP@0.5 / mAP@0.5:0.95 为 0.6875 / 0.3185；YOLOv2 项目实现为 0.6936 / 0.3941。本次结果更高，但模型主干、输入/训练配方和头部都不同，不能把差值解释成只由 YOLO 版本带来的提升。它证明的是这套 ConvNeXt-Small + YOLOv3-head 配方在该 VOC 划分上表现更好。

VOC 2007 test 已参与过此前的项目对比，因此这里的 test 结果适合作为项目内统一指标，不等同于从未触碰的外部盲测结果。

## 产物位置

运行目录：`runs/yolov3/voc0712-convnext-small-freeze10-bblr01-416-bf16-bs16-20261007/`

- `best.pt`：epoch 185 最佳验证 checkpoint；训练权重留在本地运行目录，不随源码提交。
- `test_metrics.json`：完整 test 集指标和逐类 AP。
- `training_curves.png` / `.svg`：训练损失、验证指标和学习率曲线。
- `dataset_analysis.png` / `.json`：VOC 划分与类别分布分析。
- `predictions/contact_sheet.jpg`、单图和 `manifest.json`：测试集定性结果。

![YOLOv3-head ConvNeXt-Small training curves](../../runs/yolov3/voc0712-convnext-small-freeze10-bblr01-416-bf16-bs16-20261007/training_curves.png)

![YOLOv3-head ConvNeXt-Small test predictions](../../runs/yolov3/voc0712-convnext-small-freeze10-bblr01-416-bf16-bs16-20261007/predictions/contact_sheet.jpg)
