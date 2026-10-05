# YOLOv1 历史策略与训练产物清理记录（2026-10-05）

按用户要求，归档旧实验使用的训练配方和学习结论后，删除 `runs/yolov1/` 中本表列出的 10 个历史 run 目录。当前最终模型的 run `voc0712-convnext-small-grid14-giou-final-20261005` 完整保留，包含 checkpoint、训练曲线、数据集分析、VOC test 指标和预测可视化。

历史实验的方案快照保存在 `configs/yolov1/archive/`；仓库原有的正式配方仍在 `configs/yolov1/`。各实验报告保留在 `docs/experiments/yolov1/` 与 `docs/reports/`，其中旧 run 产物链接已替换为清理说明。归档配方保留模型、输入尺寸、优化器、学习率、冻结轮次、数据增强和损失设置；复跑时按文档命令提供本地数据集路径。

| 已清理 run | 配方快照 | 保留的学习笔记 |
|---|---|---|
| `voc0712-convnext-small-grid14-coord10-20261005` | [配方](../../../configs/yolov1/archive/voc0712-convnext-small-grid14-coord10-20261005.yaml) | [coord10 基线](voc0712_convnext_small_grid14_coord10_20261005.md) |
| `voc0712-convnext-small-grid14-exposurematched-20261005` | [配方](../../../configs/yolov1/archive/voc0712-convnext-small-grid14-exposurematched-20261005.yaml) | [训练曝光量对照](voc0712_convnext_small_grid14_exposurematched_20261005.md) |
| `voc0712-convnext-small-grid14-giouaux-20261005` | [配方](../../../configs/yolov1/archive/voc0712-convnext-small-grid14-giouaux-20261005.yaml) | [GIoU 定位实验与最终重训](voc0712_convnext_small_grid14_giou_aux_20261005.md) |
| `voc2007-convnext-small-convhead-20261002` | [配方](../../../configs/yolov1/archive/voc2007-convnext-small-convhead-20261002.yaml) | [ConvNeXt Small 初始完整实验](convnext_small_voc2007_20261002.md) |
| `voc2007-convnext-small-grid14-20261004` | [配方](../../../configs/yolov1/archive/voc2007-convnext-small-grid14-20261004.yaml) | [7×7 / 14×14 对照](convnext_small_grid14_voc2007_20261004.md) |
| `voc2007-convnext-small-grid7-control-20261005` | [配方](../../../configs/yolov1/archive/voc2007-convnext-small-grid7-control-20261005.yaml) | [7×7 / 14×14 对照](convnext_small_grid14_voc2007_20261004.md) |
| `voc2007-convnext-small-lr10x-20261004` | [配方](../../../configs/yolov1/archive/voc2007-convnext-small-lr10x-20261004.yaml) | [学习率对照](convnext_small_grid14_voc2007_20261004.md) |
| `voc2007-convnext-small-lr-control-20261004` | [配方](../../../configs/yolov1/archive/voc2007-convnext-small-lr-control-20261004.yaml) | [学习率对照](convnext_small_grid14_voc2007_20261004.md) |
| `voc2007-convnext-tiny-convhead-20261001` | [主配方](../../../configs/yolov1/archive/voc2007-convnext-tiny-convhead-20261001.yaml)、[lr baseline 100](../../../configs/yolov1/archive/voc2007-convnext-tiny-convhead-20261001-lr-baseline100.yaml)、[lr baseline 70](../../../configs/yolov1/archive/voc2007-convnext-tiny-convhead-20261001-lr-baseline70.yaml) | [骨干和调度实验](../../reports/voc2007_training_strategy_20261001.md) |
| `voc2007-grid7-vs-grid14-20261005` | 训练设置见对应的 7×7 与 14×14 配方和学习笔记 | [网格对照](convnext_small_grid14_voc2007_20261004.md) |

清理只针对上表历史 `runs/yolov1/` 输出目录，不包含数据集、源码、正式配置、学习笔记和新的最终 run。
