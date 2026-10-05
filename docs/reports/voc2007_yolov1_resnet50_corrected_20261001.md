# VOC2007 YOLOv1 风格 ResNet-50：修正增强后的完整训练报告

日期：2026-10-01。运行目录：[voc2007-resnet50-corrected-20261001](../../runs/yolov1/voc2007-resnet50-corrected-20261001/)。本报告对应一次从 ImageNet ResNet-50 权重重新初始化的完整 135 轮训练，以及按验证集选定权重后的一次 VOC2007 test 评估。

## 结果与结论

| 模型与 split | 轮次 | mAP@0.50 | mAP@[0.50:0.95] | 图像数 |
| --- | ---: | ---: | ---: | ---: |
| 本次最佳 checkpoint，VOC2007 val | 130 | 0.4991 | **0.2104** | 2,510 |
| 本次最后 checkpoint，VOC2007 val | 135 | 0.4989 | 0.2094 | 2,510 |
| 本次最佳 checkpoint，VOC2007 test | 130 | **0.5177** | **0.2124** | 4,952 |
| 先前整程冻结 backbone 的 checkpoint，VOC2007 test | 135 | 0.0784 | 0.0294 | 4,952 |

第 130 轮按 val mAP@[0.50:0.95] 选为 `best.pt`，第 135 轮保存为 `last.pt`。修正后的模型已学到可辨认的大目标，较整程冻结的首轮基线明显改善；在小目标、密集目标和某些类别上仍有明显漏检。第 90～135 轮 val AP 基本进入平台，单纯增加少量 epoch 没有明显收益。它是**YOLOv1 风格迁移学习实验基线**，不能称为论文原版 YOLOv1 或官方 VOC 指标复现。

**可视化补充诊断：**固定 0.25 阈值的 20 张样例 recall 较低，降到 0.10 会找回更多目标，也产生更多误检；部分目标在 0.001 下仍漏检。逐图、后处理与论文对照见[推理可视化问题诊断](voc2007_yolov1_visualization_diagnosis_20261001.md)。

## 数据与训练条件

- 数据源：PASCAL VOC2007 官方 train/val/test，转换为派生 YOLO TXT；训练只用 train 的 2,501 张图、6,301 个非 `difficult` 框；val 有 2,510 张、6,307 框；test 有 4,952 张、12,032 框。没有混入 VOC2012，也没有用 test 选 checkpoint。
- 转换器省略 VOC XML 中 `difficult=1` 的目标，train/val/test 分别省略 1,543/1,511/2,944 个。项目评估使用 YOLO TXT 标签与连续插值 AP；因此数值不可直接与 VOC 官方 devkit AP 对比。
- 模型：ImageNet 预训练 ResNet-50 主干加 YOLOv1 风格检测头；448×448 输入、7×7 网格、每格 2 框、20 类。原论文使用不同主干，当前模型和预处理属于明确工程变体。
- 配方：[voc2007_resnet50_corrected.yaml](../../configs/yolov1/voc2007_resnet50_corrected.yaml)。前 5 轮冻结主干，第 6 轮起全部解冻；Adam，head 学习率 `1e-4`、backbone `1e-5`，第 75/105 轮各降 10 倍。micro batch 4、梯度累积到有效 batch 64、seed 42；每 10 轮周期保存，验证也每 10 轮执行，并在末轮执行。
- 训练设备为 RTX 4060 Ti。完整 run 的文件时间从 2026-09-30 23:45 至 2026-10-01 02:42，约 2 小时 57 分；这是该环境中的实际历时，不是跨硬件的速度承诺。
- 此次从干净 ImageNet 权重启动。上一份在错误仿射增强下训练到第 114 轮的 checkpoint 没有用于初始化；诊断见[停训报告](voc2007_yolov1_resnet50_finetune_stopped_20260930.md)。

## 数据集分析

[数据集分析图](../../runs/yolov1/voc2007-resnet50-corrected-20261001/dataset_analysis.png)（[可缩放 SVG](../../runs/yolov1/voc2007-resnet50-corrected-20261001/dataset_analysis.svg)；[原始统计 JSON](../../runs/yolov1/voc2007-resnet50-corrected-20261001/dataset_analysis.json)）覆盖各 split 的类别计数、每图目标数、宽高、中心分布、面积与 7×7 网格冲突。

train 中 `person` 有 2,358 框，占 6,301 框的 37.4%。train/val/test 分别有 289/294/558 张图片发生同格中心冲突，同格额外框分别为 421/501/823。当前 YOLOv1 训练目标每格只保留一个对象，train 的 421 个额外框无法进入网格监督；评估仍使用全部保留的真值框。train 框面积占整图面积的中位数约 10.0%，第 10 百分位约 1.07%。这些统计解释了部分密集或较小目标的难度，但不能单独证明误差成因。

## 曲线与逐类结果

- [训练曲线 PNG](../../runs/yolov1/voc2007-resnet50-corrected-20261001/training_curves.png)（[SVG](../../runs/yolov1/voc2007-resnet50-corrected-20261001/training_curves.svg)；[逐轮 JSONL](../../runs/yolov1/voc2007-resnet50-corrected-20261001/metrics.jsonl)）：总损失从第 1 轮 12.524 降到第 135 轮 1.561；图中同时有坐标、置信度、分类损失，val mAP、逐类 AP、学习率和每轮忽略目标数。第 30 轮 val mAP@0.50/0.50:0.95 为 0.1689/0.0606，第 90 轮为 0.4941/0.2004；后期增益有限。
- [test 逐类 AP 图](../../runs/yolov1/voc2007-resnet50-corrected-20261001/test_per_class_ap.png)（[SVG](../../runs/yolov1/voc2007-resnet50-corrected-20261001/test_per_class_ap.svg)；[测试结果 JSON](../../runs/yolov1/voc2007-resnet50-corrected-20261001/heldout_test_best.json)）：AP@0.50 较高的有 `cat` 0.788、`dog` 0.769、`horse` 0.763、`train` 0.721；较低的有 `pottedplant` 0.147、`bottle` 0.156、`chair` 0.211、`boat` 0.262。
- test 输出 489,846 个检测，即平均 98.9 个/图，接近配置的每图 100 框上限。该数字说明低阈值 AP 评估可能受截断影响；尚未做不同上限的对照，因此不把它断定为 AP 较低的唯一原因。

## 推理可视化

[20 类样例总览](../../runs/yolov1/voc2007-resnet50-corrected-20261001/predictions/contact_sheet.jpg)从 test 中按固定随机种子 42 为每类选择一张**不重复**图片。总览和每张完整图都按“左侧真值、右侧预测”展示，例如 [dog/person](../../runs/yolov1/voc2007-resnet50-corrected-20261001/predictions/11_dog_007181.jpg)、[bottle](../../runs/yolov1/voc2007-resnet50-corrected-20261001/predictions/04_bottle_002231.jpg)、[chair](../../runs/yolov1/voc2007-resnet50-corrected-20261001/predictions/08_chair_000940.jpg)。[manifest.json](../../runs/yolov1/voc2007-resnet50-corrected-20261001/predictions/manifest.json)列出 20 张的来源、真值和预测坐标。

可视化使用 confidence ≥ 0.25、class-aware NMS IoU 0.45、最多 100 框/图。图中的得分阈值高于 AP 评估使用的 0.001，因此“图上没画框”不等于该类别在 AP 评估中完全没有候选。样例没有按预测成功率挑选：`dog/person` 两个目标都检出，`bottle` 图仅检出多个瓶子中的一部分，`diningtable`、`pottedplant` 等图有漏检。

## 已执行检查与复现入口

修正的图像/框几何对齐回归测试已通过；`python -m pytest -q` 共 8 项通过。训练记录完整覆盖 epoch 1～135、84,375 次 micro batch 迭代；最多每 16 次 micro batch 累积一次梯度更新，epoch 末不足 16 次也执行更新。best/last checkpoint、曲线、数据分析、test JSON 和 20 张对照图均已落盘。已打开检查训练曲线、数据分析图、逐类 AP 图、总览和单张真值/预测图。

从仓库根目录重新生成图表或查看权重可用：

```bash
RUN=runs/yolov1/voc2007-resnet50-corrected-20261001
VOC=/path/to/yolo_voc2007
python scripts/plot_training_curves.py --run-dir "$RUN"
python scripts/plot_dataset_analysis.py --config configs/datasets/voc2007.yaml --dataset-root "$VOC" --output-prefix "$RUN/dataset_analysis"
python scripts/plot_evaluation_metrics.py --report "$RUN/heldout_test_best.json" --output-prefix "$RUN/test_per_class_ap"
python scripts/visualize_yolov1_predictions.py --recipe "$RUN/train_recipe.yaml" --dataset-root "$VOC" --checkpoint "$RUN/best.pt" --output-dir "$RUN/predictions" --split test --confidence 0.25 --nms-iou 0.45 --seed 42 --device cuda
```

`heldout_test_best.json` 和可视化 manifest 中的 checkpoint 路径记录的是实际运行时的临时高速盘位置；相同权重已复制到上述 `RUN/best.pt`。重新运行完整 test 可使用 `scripts/evaluate_yolov1.py --dataset-root "$VOC" --recipe "$RUN/train_recipe.yaml" --checkpoint "$RUN/best.pt" --split test --device cuda --output <新的结果文件>`；本报告中的 test 数值来自已执行的一次评估。
