# ConvNeXt Small on VOC2007
> 清理记录（2026-10-05）：旧 run 的 checkpoint、逐轮日志和图像产物已按请求清理；实验方案、主要指标和结论保留在本文，配方快照见[历史策略归档](historical_strategy_archive_20261005.md)。


## 问题与假设

前一轮 ConvNeXt Tiny 135 轮 run 的 test mAP50 为 `0.5668`，mAP50:95 为 `0.2113`；高 IoU 定位与可视化仍不理想。本实验检查 TorchVision ConvNeXt Small 是否在相同 YOLOv1 卷积头和训练配方下带来验证和测试收益。假设是更深的特征主干可能改善定位；这需要用相同 split 的 AP50 与 AP50:95、训练曲线和预测图判断。

## 基线与固定条件

基线是 [`voc2007_convnext_tiny_convhead.yaml`](../../../configs/yolov1/voc2007_convnext_tiny_convhead.yaml) 和 [Tiny 完整训练报告](../../reports/voc2007_training_strategy_20261001.md)。新配方仅将 `backbone_name` / ImageNet 权重来源替换为 ConvNeXt Small；其余使用相同 VOC2007 train/val/test、448 直接缩放、7×7 输出、卷积检测头、冻结 5 轮、Adam、head LR `1e-4`、backbone LR `1e-5`、有效 batch 64、随机种子 42、增强、75/105 轮 LR 衰减和每 10 轮检查点更新。

为了和 Tiny 训练运行保持一致，前 30 轮使用 FP32，从第 31 轮恢复 `last.pt` 后使用 BF16 训练前向；loss 与验证仍为 FP32。TorchVision 官方 `IMAGENET1K_V1` ConvNeXt Small 权重约 191.7 MB，参数约 50.2M，官方分类模型基准为 8.68 GFLOPs（224 输入）；此 GFLOPs 不是本项目 448 输入检测模型的实测值。[官方模型与权重说明](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_small.html)。

代码将该模型接入现有 ConvNeXt feature 路径，448 输入预期产出 `[N,768,14,14]` 特征，卷积头输出 `[N,7,7,30]`。YOLOv1 标签分配、损失和解码均不变；这是 ImageNet 迁移的工程变体，不是论文原版 YOLOv1。

## 运行命令

```bash
python scripts/train_yolov1.py \
  --dataset-root /path/to/yolo_voc2007 \
  --recipe configs/yolov1/voc2007_convnext_small_convhead.yaml \
  --output runs/yolov1/voc2007-convnext-small-convhead-20261002 \
  --device cuda:0 --epochs 30

python scripts/train_yolov1.py \
  --dataset-root /path/to/yolo_voc2007 \
  --recipe configs/yolov1/voc2007_convnext_small_convhead.yaml \
  --output runs/yolov1/voc2007-convnext-small-convhead-20261002 \
  --device cuda:0 --precision bf16 \
  --resume runs/yolov1/voc2007-convnext-small-convhead-20261002/last.pt
```

训练后可复现测试、曲线、数据集分析和固定 seed 的 test 拼图：

```bash
python scripts/evaluate_yolov1.py \
  --dataset-root /path/to/yolo_voc2007 \
  --recipe configs/yolov1/voc2007_convnext_small_convhead.yaml \
  --checkpoint runs/yolov1/voc2007-convnext-small-convhead-20261002/best.pt \
  --split test --device cuda:0 --batch-size 4 \
  --output runs/yolov1/voc2007-convnext-small-convhead-20261002/test_metrics.json

python scripts/plot_training_curves.py \
  --run-dir runs/yolov1/voc2007-convnext-small-convhead-20261002

python scripts/plot_dataset_analysis.py \
  --config configs/datasets/voc2007.yaml \
  --dataset-root /path/to/yolo_voc2007 \
  --output-prefix runs/yolov1/voc2007-convnext-small-convhead-20261002/dataset_analysis

python scripts/visualize_yolov1_predictions.py \
  --recipe configs/yolov1/voc2007_convnext_small_convhead.yaml \
  --dataset-root /path/to/yolo_voc2007 \
  --checkpoint runs/yolov1/voc2007-convnext-small-convhead-20261002/best.pt \
  --output-dir runs/yolov1/voc2007-convnext-small-convhead-20261002/predictions_test_conf025 \
  --split test --confidence 0.25 --nms-iou 0.45 --seed 42 --device cuda:0
```

## 结果

### 训练与验证

训练共完成 135 个有效 epoch。前 30 轮为 FP32；第 31 轮起使用 BF16 训练前向，loss 与验证仍为 FP32。前 5 轮冻结 backbone，此后全量微调。运行中在 epoch28 后曾被中断；当时最新可恢复 checkpoint 是 epoch20，因此从该 checkpoint 重跑 epoch21–30，再继续 BF16 阶段。中断前的 epoch1–28 指标保存在 `metrics_interrupted_epoch28_20261002.jsonl`；正式曲线使用与恢复权重连续一致的 `metrics.jsonl`（epoch1–135，无重复轮次）。未持久化权重的 epoch21–28 计算被重跑。

最佳验证点为 **epoch130**：mAP50 `0.60554`、mAP50:95 `0.25211`。最终 epoch135 为 `0.60731 / 0.25181`，与最佳点接近。epoch80 的首次学习率衰减后 mAP50:95 升到 `0.24766`；epoch100、110、120、130 分别为 `0.25196 / 0.25105 / 0.25080 / 0.25211`，后段基本进入平台。训练总损失从 epoch1 的 `10.39` 降至 epoch135 的 `1.09`，后段没有发散迹象。

实际解冻阶段 FP32 约需 120 秒/轮；BF16 实测约 43–47 秒/轮。减速点与 epoch6 解冻 backbone 一致，BF16 后吞吐约提高 2.5 倍。环境为 GeForce RTX 4060 Ti 16 GB、PyTorch 2.10.0、TorchVision 0.25.0、CUDA 12.8。

### Test 结果与 Tiny 对照

使用验证集选出的 epoch130 `best.pt`，在 VOC2007 test 的 4,952 张图上评估。指标置信度下限为 `0.001`、NMS IoU `0.45`、每图最多 100 个检测框。

| 模型 | test mAP50 | test mAP50:95 |
| --- | ---: | ---: |
| ConvNeXt Tiny，epoch100 best | 0.56684 | 0.21128 |
| ConvNeXt Small，epoch130 best | **0.62669** | **0.25707** |
| 绝对变化 | +0.05985 | +0.04578 |

Small 在 20 个类别的 AP50 与 AP50:95 都高于 Tiny。整体 mAP50 相对提高约 10.6%，mAP50:95 相对提高约 21.7%。这支持继续采用 ConvNeXt Small 做当前候选，但提升应解释为“backbone 与各自 ImageNet 预训练权重”的组合差异；它不是只改变网络拓扑的严格消融，也不是 YOLOv1 论文原始配置的严格复现。

Small 的 test 弱项仍集中在小物体和定位：pottedplant 的 AP50 / AP50:95 为 `0.255 / 0.060`，bottle 为 `0.231 / 0.064`，chair 为 `0.401 / 0.123`，boat 为 `0.436 / 0.130`。这几类虽然相对 Tiny 有提升，绝对 AP50:95 仍低。test 评估置信度下限为 `0.001`，所以共记录 491,218 个候选检测（约 99 个/图，接近 100 框上限）；这用于计算 AP，不代表实际应用应输出这么多框，部署阈值仍需在 validation split 上单独确定。

### 数据分析与预测检查

| Split | 图像 | 标注框 | 有同格中心冲突的图像 | 同格额外目标 |
| --- | ---: | ---: | ---: | ---: |
| train | 2,501 | 6,301 | 289（11.6%） | 421（6.7%） |
| val | 2,510 | 6,307 | 294（11.7%） | 501（7.9%） |
| test | 4,952 | 12,032 | 558（11.3%） | 823（6.8%） |

扫描没有发现空标签文件。train 中 person 有 2,358 个框（约 37.4%），diningtable 只有 103 个，呈长尾分布。train 框面积的第 10、50、90 百分位分别约为图像面积的 `1.07% / 9.98% / 55.15%`。7×7 网格冲突和小物体会限制 YOLOv1 单格目标表达能力；这是数据与模型结构的已知约束，不能只靠换 backbone 消除。

预测拼图从 test split 用固定 seed 42 每类选 1 张含该类标注的图，展示的是定性样例，不是总体召回统计。置信度 `0.25` 时，易样例的框坐标和尺寸与 GT 对齐；拥挤 bottle 样例检出 5/7 个 GT，pottedplant 3/8，car 4/7，diningtable 和 train 示例未输出对应类别框。把显示阈值降到 `0.10` 会出现更多弱分数框，也带来更多误检和重叠。可视化坐标映射在简单样例中正确；当前剩余问题主要是漏检、混类和高 IoU 定位不足，不是拼图缩放导致的整体坐标偏移。样例只用于定位问题，不以单张图代替汇总指标。

### 产物

- 最佳权重：best.pt（历史 run 产物已清理）（epoch130）；最终权重：last.pt（历史 run 产物已清理）（epoch135）。
- 完整逐轮指标：metrics.jsonl（历史 run 产物已清理）；test 逐类指标：test_metrics.json（历史 run 产物已清理）。
- 训练与验证曲线：PNG（历史 run 产物已清理）、SVG（历史 run 产物已清理）。
- 数据集分析：PNG（历史 run 产物已清理）、JSON（历史 run 产物已清理）。
- Test 预测拼图：置信度 0.25（历史 run 产物已清理）、置信度 0.10（历史 run 产物已清理），逐图结果与标注在对应目录的 `manifest.json`。

训练、曲线、数据集图和两组拼图均已实际生成。`python -m pytest -q` 结果为 `8 passed`。VOC2007 test 图像此前已用于视觉检查，所以这里是同数据协议下的训练外测试对比，不称作完全盲测。

### 判断与下一步

ConvNeXt Small 相比 Tiny 的 test mAP50 与 mAP50:95 都提高，且验证最佳值高于 Tiny；当前它是两者中更好的候选。结果比之前更可用，但密集/小物体漏检和定位误差仍明显，不能据此称为可直接部署的通用检测器。下一步应在 validation split 上按类别检查置信度和错误样例，优先针对 bottle、pottedplant 以及 7×7 同格冲突设计单变量实验；test 指标保留作最终对比，不用于选择阈值或继续调参。
