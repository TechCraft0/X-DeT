# YOLOv2 完整训练与 YOLOv1 对比

**运行日期：** 2026-10-05　 **数据集：** VOC 2007 train + VOC 2012 trainval，VOC 2007 test 评估　 **结果：** 完成 165/165 epochs

## 结果摘要

YOLOv2 在同一 VOC 2007 test split 上的 mAP50 比 YOLOv1 高 `0.61` 个百分点，mAP50:95 高 `7.56` 个百分点。提升主要出现在更严格的 IoU 阈值下，与框定位更贴合的表现一致；但这不是单变量结构消融，两个版本使用了各自不同的骨干和训练配方，不能把全部增益单独归因于 YOLOv2 的 anchor 结构。

| 模型 | 本次对比所用实现 | 输入 | Test mAP50 | Test mAP50:95 |
|---|---|---:|---:|---:|
| YOLOv1 | ConvNeXt-Small，14×14 网格，GIoU 训练变体 | 448×448 | 0.6875 | 0.3185 |
| YOLOv2 | Darknet-19，5 anchors，passthrough | 416×416 | **0.6936** | **0.3941** |
| YOLOv2 − YOLOv1 | 同 test split、同评估器 | — | **+0.0061** | **+0.0756** |

## 可用权重

- [推理权重 `best_inference.pt`](../../runs/yolov2/voc0712-darknet19-pretrained-20261005/best_inference.pt)：约 193 MiB，仅含模型参数和来源元信息，已经用单图推理和完整 test 评估验证。SHA256：`f4101857c29a7ea47fdc5f233c7a982b98d0853983cac4835b6630521b449734`。
- [完整最佳 checkpoint `best.pt`](../../runs/yolov2/voc0712-darknet19-pretrained-20261005/best.pt)：含模型与优化器，可做评估或恢复训练；best epoch 为 165。
- [最终恢复点 `last.pt`](../../runs/yolov2/voc0712-darknet19-pretrained-20261005/last.pt)：第 165 epoch 的最终训练状态。

这些权重位于本机 `runs/` 目录并被 `.gitignore` 忽略，不会随代码提交。直接推理示例：

```bash
python scripts/infer_yolov2.py path/to/image.jpg \
  --checkpoint runs/yolov2/voc0712-darknet19-pretrained-20261005/best_inference.pt \
  --recipe configs/yolov2/voc0712.yaml --device cuda
```

## 数据、权重来源与训练设置

数据使用仓库 VOC trainmix 配置，类别为 VOC 20 类，标签是 YOLO TXT。原始数据只读，未改动。

| Split | 图片 | 框标注 | 空标签文件 |
|---|---:|---:|---:|
| Train（VOC07 train + VOC12 trainval） | 14,041 | 33,751 | 0 |
| Validation（VOC07 val） | 2,510 | 6,307 | 0 |
| Test（VOC07 test） | 4,952 | 12,032 | 0 |

训练集类别数量不均衡：person 有 10,924 个框，sofa 有 690 个，约相差 15.8 倍。YOLOv2 的 13×13 网格中心碰撞统计为 train 622 张图 / 764 个额外目标、val 109 / 136、test 198 / 244；此统计只说明目标中心落在相同网格，不等于最终的 cell/anchor 分配冲突。

初始化文件是 [Darknet 官方 ImageNet Darknet-19 448 分类权重](https://pjreddie.com/media/files/darknet19_448.weights)，SHA256：`77bd0b33f92522a97d6667c5d6cb118d4928bec40f21872f5b4965231ac2167b`。训练时加载前 18 个 Darknet-19 特征卷积块及 BatchNorm 状态，分类输出层不加载；检测专属层随机初始化。它是分类 backbone 权重，不是官方 YOLOv2 检测器 checkpoint。

| 项目 | 实际设置 |
|---|---|
| GPU / 软件 | NVIDIA GeForce RTX 4060 Ti；Python 3.10.12；PyTorch 2.10.0+cu128；CUDA 12.8 |
| 训练 | 165 epochs；输入 416；BF16 autocast；micro-batch 4，梯度累积 16，有效 batch 64；seed 42 |
| 优化器 | SGD；初始 LR 0.001，momentum 0.9，weight decay 0.0005；第 146、164 epoch 衰减 10 倍 |
| 验证与 checkpoint | 每 5 epochs 验证；按 val mAP50:95 保存 best；每 10 epochs 保存恢复点，训练结束额外保存最终恢复点 |
| 训练损失 | epoch 1 为 8.973；epoch 165 为 0.408；最终学习率 `1e-5` |

复跑命令（`DATASET_ROOT` 指向相同的数据目录；已有 run 不会被覆盖）：

```bash
python scripts/train_yolov2.py \
  --dataset-root "$DATASET_ROOT" \
  --recipe configs/yolov2/voc0712.yaml \
  --pretrained-darknet weights/yolov2/darknet19_448.weights \
  --device cuda:0 --precision bf16 --micro-batch-size 4 \
  --output runs/yolov2/voc0712-darknet19-pretrained-rerun
```

## 完整评估结果

验证集最终指标：mAP50 `0.6853`、mAP50:95 `0.3921`，2,510 张图。test 指标来自 best checkpoint，在 4,952 张图上评估；confidence 阈值 `0.001`、NMS IoU `0.45`、每图最多 100 个检测框。

复核两版 test 指标的命令：

```bash
python scripts/evaluate_yolov1.py \
  --dataset-root "$DATASET_ROOT" \
  --recipe configs/yolov1/voc0712_convnext_small_grid14_giou_aux.yaml \
  --checkpoint runs/yolov1/voc0712-convnext-small-grid14-giou-final-20261005/best.pt \
  --split test --device cuda:0 --batch-size 4 \
  --output runs/yolov2/voc0712-darknet19-pretrained-20261005/yolov1_baseline_test.json

python scripts/evaluate_yolov2.py \
  --dataset-root "$DATASET_ROOT" \
  --recipe configs/yolov2/voc0712.yaml \
  --checkpoint runs/yolov2/voc0712-darknet19-pretrained-20261005/best_inference.pt \
  --split test --device cuda:0 --batch-size 4 \
  --output runs/yolov2/voc0712-darknet19-pretrained-20261005/test_metrics_inference_checkpoint.json
```

| Test 指标 | YOLOv1 | YOLOv2 |
|---|---:|---:|
| mAP50 | 0.6875 | 0.6936 |
| mAP50:95 | 0.3185 | 0.3941 |
| 图片数 | 4,952 | 4,952 |
| 评估后检测框数 | 494,017 | 239,454 |

逐类 AP50:95（AP 为 0–1 比例，差值为 YOLOv2 − YOLOv1）：

| 类别 | YOLOv1 | YOLOv2 | 差值 |
|---|---:|---:|---:|
| aeroplane | 0.338 | 0.439 | +0.101 |
| bicycle | 0.353 | 0.450 | +0.097 |
| bird | 0.304 | 0.344 | +0.040 |
| boat | 0.158 | 0.287 | +0.130 |
| bottle | 0.164 | 0.180 | +0.016 |
| bus | 0.439 | 0.553 | +0.114 |
| car | 0.391 | 0.523 | +0.132 |
| cat | 0.445 | 0.519 | +0.074 |
| chair | 0.194 | 0.195 | +0.001 |
| cow | 0.417 | 0.384 | −0.032 |
| diningtable | 0.161 | 0.375 | +0.214 |
| dog | 0.442 | 0.464 | +0.022 |
| horse | 0.403 | 0.471 | +0.067 |
| motorbike | 0.361 | 0.455 | +0.095 |
| person | 0.300 | 0.375 | +0.075 |
| pottedplant | 0.105 | 0.142 | +0.037 |
| sheep | 0.386 | 0.384 | −0.002 |
| sofa | 0.311 | 0.432 | +0.121 |
| train | 0.357 | 0.500 | +0.143 |
| tvmonitor | 0.340 | 0.409 | +0.069 |

18/20 类的 AP50:95 提高；增幅较大的类别为 diningtable、train、car、boat 和 sofa。cow 的 AP50:95 下降 0.032，sheep 下降 0.002；pottedplant、bottle、chair 的最终 AP50:95 仍低于 0.20，是后续应重点检查的类别。数据类别不均衡可能是因素之一，但单次对比不足以证明因果。

## 图表和预测可视化

以下运行产物在 `runs/yolov2/voc0712-darknet19-pretrained-20261005/` 下，均由本次实验生成且不纳入 Git：

| 内容 | 文件 |
|---|---|
| 训练损失、验证指标和学习率曲线 | [training_curves.png](../../runs/yolov2/voc0712-darknet19-pretrained-20261005/training_curves.png) |
| 数据集类别、框尺寸与位置分析 | [dataset_analysis.png](../../runs/yolov2/voc0712-darknet19-pretrained-20261005/dataset_analysis.png) |
| V1/V2 训练与验证指标对比 | [training_comparison.png](../../runs/yolov2/voc0712-darknet19-pretrained-20261005/training_comparison.png) |
| Test 逐类 AP50:95 对比 | [test_class_ap_comparison.png](../../runs/yolov2/voc0712-darknet19-pretrained-20261005/test_class_ap_comparison.png) |
| 同 20 张图的真值 / YOLOv1 / YOLOv2 预测 | [prediction_comparison_test.jpg](../../runs/yolov2/voc0712-darknet19-pretrained-20261005/prediction_comparison_test.jpg) |
| Test 逐类原始指标 | [test_comparison_metrics.json](../../runs/yolov2/voc0712-darknet19-pretrained-20261005/test_comparison_metrics.json) |
| 精简推理权重的完整 test 指标 | [test_metrics_inference_checkpoint.json](../../runs/yolov2/voc0712-darknet19-pretrained-20261005/test_metrics_inference_checkpoint.json) |

预测拼图用 seed 42 从每个类别选取一张 VOC2007 test 图片，三列分别为 Ground truth、YOLOv1、YOLOv2；可视化置信度阈值为 `0.25`，与计算 AP 用的 `0.001` 不同。该图用于查看样例，不替代全量指标。

后续又从 epoch165 以 `1e-5` 学习率续训至 epoch195。验证集最高 mAP50:95 仍是原 epoch165 的 `0.39206`；延长训练的实验过程和结论见[低学习率续训记录](../experiments/yolov2/voc0712_low_lr_continuation_20261006.md)及[完整训练曲线](../../runs/yolov2/voc0712-darknet19-pretrained-continue-lr1e5-30ep-20261006/training_curves.png)。因此报告中的 test 分数和推荐推理权重仍对应原始 epoch165 模型。

## 对比边界与后续方向

- 两个版本使用相同数据集定义、test split、检测评估函数、AP 阈值和每图检测数上限；比较回答的是“当前两套完整 recipe 在此 VOC 设置下的结果”。
- 这不是只改 backbone 或只改检测头的严格结构消融：YOLOv1 使用 ConvNeXt-Small、448 输入、14×14 网格及 GIoU 训练变体；YOLOv2 使用 Darknet-19 ImageNet 分类初始化、416 输入、13×13 网格及 anchor 检测头。
- VOC2007 test 曾用于之前的 YOLOv1 开发对比，因此这里不是完全未触碰的盲测集。若用于正式论文结论，应另留独立数据集或重新划定最终 holdout。
- 当前实现没有 YOLO9000 WordTree 联合检测/分类训练、论文中的动态多尺度训练调度或设备后端验证；所以本结果是完成训练和评估的教学型 PyTorch YOLOv2 版本，不应称为逐项严格复现或部署验证。
- 延长到 epoch195 的低学习率实验没有超过 epoch165 的验证 mAP50:95；后续优先单独测试 512 输入微调，再决定是否检查 anchors 或定位损失。每轮只改一个主要变量。
