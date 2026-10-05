# YOLOv1 ConvNeXt Small：GIoU 辅助定位实验

## 问题与假设

上一轮将 `lambda_coord` 从 5 调到 10 后，VOC2007 test 的 mAP50:95 从 `0.28683` 升至 `0.28911`，但 mAP50 从 `0.67084` 降至 `0.66679`。固定样例显示框中心整体偏差较小、宽高边界偏松。原平方误差定位项不直接优化最终 IoU。

假设：在保留坐标 MSE 作为稳定项的前提下，增加 GIoU 目标能让框边界更贴合真值。GIoU 适用于无重叠框；坐标 MSE 保留是为了继续把宽高平方根参数拉回正值区域，避免仅使用经 `clamp_min(0)` 的宽高解码时，对负输出产生零梯度。

这是 YOLOv1 风格工程变体，不是原论文严格复现。

## 单变量设计

对照基线为 `lambda_coord=10` 的 ConvNeXt Small 14×14 配方。唯一训练目标改动是 `lambda_giou` 从默认 `0` 设为 `1.0`；数据、初始化来源、种子、主干、学习率、训练轮数、batch、增强、精度、验证和推理后处理均保持不变。

新增项为负责框上的 `lambda_giou * sum(1 - GIoU) / batch_size`。负责预测框仍按 detached IoU 选择；置信度目标仍为 IoU。新增的 GIoU 保留梯度，`lambda_giou=0` 时默认配方的数学目标不变。GIoU 系数不是从论文或既有 benchmark 得出的最优值，先用 `1.0` 做单变量验证。

开始全量运行前完成了：

- 定向检查：完全重合框的 GIoU 损失为零，错位框的 GIoU 为正且对中心输出有有限梯度。
- 真实模型和真实训练 batch 的 forward/backward 检查：输入 4 张图，输出形状 `[4,14,14,30]`，各损失分项和检测头梯度均有限。既有 coord10 checkpoint 的 8 个 microbatch 上，坐标项均值为 `2.2457`、`lambda_giou=1` 项均值为 `0.7350`，新项有实际权重但没有压过原坐标项。
- 全量单元测试：`python -m pytest -q`，`10 passed`。

## 配置和产物

- 配方：[voc0712_convnext_small_grid14_giou_aux.yaml](../../../configs/yolov1/voc0712_convnext_small_grid14_giou_aux.yaml)
- 当前可交付运行：[final run](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005)
- 基线运行：[lambda_coord=10](../../../docs/experiments/yolov1/voc0712_convnext_small_grid14_coord10_20261005.md)

## 训练与验证结果

24 轮训练顺利完成，按验证集 mAP50:95 选择的最佳 checkpoint 为 epoch 24。训练末期总损失为 `4.3348`，其中坐标 MSE 为 `2.4542`、GIoU 项为 `0.6930`；学习率已衰减到 head `1e-6`、backbone `1e-7`。训练过程没有出现非有限损失。epoch 22 到 24 的验证 mAP50:95 从 `0.32614` 到 `0.32673`，增幅较小，说明本配方后段已接近平台。

| 验证集最佳结果 | mAP50 | mAP50:95 |
| --- | ---: | ---: |
| `lambda_coord=10` 对照 | 0.67712 | 0.29702 |
| GIoU 辅助项 `lambda_giou=1` | **0.69163** | **0.32673** |
| 差值 | +0.01451 | +0.02971 |

## VOC2007 test 结果

两次运行均评估 VOC2007 test 的 4,952 张图，输入缩放、后处理和每图最多 100 个检测框相同。checkpoint 都按各自验证集 mAP50:95 选择。

| 配方 | mAP50 | mAP50:95 | 检测数 |
| --- | ---: | ---: | ---: |
| `lambda_coord=5` | 0.67084 | 0.28683 | 491,997 |
| `lambda_coord=10` | 0.66679 | 0.28911 | 494,170 |
| `lambda_coord=10, lambda_giou=1` | **0.68754** | **0.31846** | 494,017 |
| GIoU 相对 coord10 差值 | +0.02075 | **+0.02935** | -153 |

相对直接对照 coord10，20 个类别的 AP50:95 全部提高；提升较大的类别是 bus `+0.05569`、cow `+0.05559`、aeroplane `+0.04916`，最小的 chair 也提高 `+0.01076`。这与“定位目标更贴近高 IoU 指标”相符，且 AP50 没有出现回退。

固定 seed 42、置信度 `0.25`、NMS IoU `0.45` 的 20 张逐类样例用于检查实际框。55 个真值中，coord10 有 36 个同类预测候选，GIoU 有 35 个；其中 33 个真值被两边模型都以 IoU≥0.5 的同类框检出。以下定位统计只计算这 33 个共同真阳性，避免把低 IoU 候选当成正确检测：

| 共同匹配框统计 | coord10 | GIoU 辅助 |
| --- | ---: | ---: |
| 中位 IoU | 0.7525 | **0.7805** |
| 中位真值覆盖率 | 0.8816 | **0.8909** |
| 中位预测面积 / 真值面积 | 1.0291 | **0.9892** |
| 中位中心偏差 / 真值对角线 | 0.03782 | 0.03726 |

共同检出的真值上，中心偏差相近，IoU 和覆盖率提高，预测框面积更接近真值。完整 55 个真值中 IoU≥0.5 的样本检出数是 coord10 `34/55`、GIoU `33/55`，所以这组小样本支持“命中后的边界拟合改善”，但不支持召回改善。样本只用于诊断，完整性能仍看 VOC test AP。

## 数据和限制

使用的数据集是 VOC2007 train + VOC2012 trainval，共 14,041 张训练图、33,751 个训练框；VOC2007 val 有 2,510 张图、6,307 个框，test 有 4,952 张图、12,032 个框。14×14 标签网格下，训练集静态统计有 506 张图存在同格目标，合计 616 个额外目标不能进入每格单目标监督；验证和测试分别有 123、180 个同格额外目标。数据集分布和框面积统计沿用本轮基线的分析结果。

GIoU 项改善了 test 上高 IoU AP 和固定匹配框质量，是目前针对框偏松问题最有证据的修正。它不解决 YOLOv1 单网格目标分配造成的密集目标漏检，训练配方也只有 24 轮。VOC2007 test 曾用于前序配方对比和定位分析，因此这里是相同 test split 上的受控项目比较，不是全新盲测；单次 seed 也不能证明对随机种子稳定。`lambda_giou=1` 是一个有效候选值，尚不能称为最优值。

## 最终重训与复现检查

为得到独立交付目录，使用同一配方从 ImageNet 预训练权重重新训练 24 轮，输出到 `runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005`。不是接续旧 checkpoint。新 run 的 168/168 个逐轮训练标量、24/24 个验证 mAP 值和完整 test 指标都与上一轮 GIoU 训练完全一致；复现摘要见 [`retraining_reproducibility.json`](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/retraining_reproducibility.json)。这确认了方案和本次环境下的训练结果可稳定重现。

最终使用新目录的 `best.pt`，验证集最佳点为 epoch 24（mAP50 `0.69163`，mAP50:95 `0.32673`）；VOC2007 test 为 mAP50 `0.68754`、mAP50:95 `0.31846`。该值与前次 GIoU run 一致；VOC test 曾用于前序方案比较，不是新盲测。

## 产物

- [新 run 训练曲线 PNG](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/training_curves.png) / [SVG](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/training_curves.svg)
- [VOC test 逐类 AP 图](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/test_per_class.png) 与[相对 coord10 的 AP50:95 差值图](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/test_ap5095_delta_vs_coord10.png) / [SVG](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/test_ap5095_delta_vs_coord10.svg)
- [最终 checkpoint](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/best.pt) 和 [test 预测拼图](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/predictions_test_conf025/contact_sheet.jpg)
- [同样例定位对照图](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/localization_sample_comparison.png) / [统计 JSON](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/localization_sample_comparison.json)
- [VOC test 指标 JSON](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/test_metrics.json)、[逐类对照 JSON](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/test_comparison.json)
- [VOC trainmix 数据分析图](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/dataset_analysis.png) / [分析 JSON](../../../runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/dataset_analysis.json)

下一步可在保持数据、网络、种子和其他训练参数不变的条件下，把 `lambda_giou` 单独调到 `0.5` 或 `2.0` 做系数对照；比较验证 mAP50:95 后只对候选方案运行一次 test，避免继续用 test 选择系数。

## 复现命令

```bash
python scripts/train_yolov1.py \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --recipe configs/yolov1/voc0712_convnext_small_grid14_giou_aux.yaml \
  --output runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005 \
  --device cuda:0

python scripts/evaluate_yolov1.py \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --recipe configs/yolov1/voc0712_convnext_small_grid14_giou_aux.yaml \
  --checkpoint runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/best.pt \
  --split test --device cuda:0 --batch-size 4 \
  --output runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/test_metrics.json

python scripts/plot_training_curves.py \
  --run-dir runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005
```
