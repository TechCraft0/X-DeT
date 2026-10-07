# YOLOv3 检测器

## 来源与范围

- 论文：Joseph Redmon、Ali Farhadi，[*YOLOv3: An Incremental Improvement*](https://arxiv.org/abs/1804.02767)，2018。
- 工程对照：[Darknet YOLOv3 VOC 配置](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-voc.cfg)、[Darknet YOLO 层](https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c)、[Darknet detector 训练与验证入口](https://github.com/pjreddie/darknet/blob/master/examples/detector.c)、[Darknet 数据增强](https://github.com/pjreddie/darknet/blob/master/src/data.c)。
- 骨干预训练权重：[darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)。使用时由训练命令显式指定，不会自动下载。
- 预训练方式说明：[官方 YOLOv3 训练页面](https://pjreddie.com/darknet/yolo/)明确说明该卷积权重来自 ImageNet 分类预训练。
- 工程变体的图像分类预训练：[TorchVision ConvNeXt-Small](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_small.html)，使用 `ConvNeXt_Small_Weights.IMAGENET1K_V1`。

本目录默认实现论文检测器的教学型 PyTorch 路径：Darknet-53、三尺度检测头、九个 anchor、跨尺度最佳 anchor 分配、objectness ignore 规则、损失和 NMS。另提供 ConvNeXt-Small 替换骨干的工程变体，用于检验大型分类预训练主干对同一 YOLOv3 检测头的影响。没有实现 Darknet 的完整训练运行时、随机多尺度训练或 COCO/VOC 官方训练代码，因此不称为逐项严格复现。

## 相对 YOLOv2 的变化

YOLOv3 保留 anchor-based 检测，但将 Darknet-19 主干换成 Darknet-53，并使用自顶向下的特征融合在 stride 32、16、8 三个尺度预测。较高分辨率的检测头使网络能在较细网格上预测小目标。类别使用相互独立的 logistic 输出，因此一个框的类别概率不经过互斥 softmax。YOLOv3 论文明确保留 YOLOv2 式 anchor box 回归，并使用 sigmoid 中心偏移、objectness 与类别 logistic 输出。[论文](https://arxiv.org/abs/1804.02767)

## 结构与张量流

默认输入为 `[N,3,416,416]`，RGB 像素除以 255。DarknetConv 是 `Conv2d → BatchNorm2d → LeakyReLU(0.1)`；残差块是 `1×1 降维 → 3×3 升维 → shortcut 相加`。

| 路径 | 输出（416 输入） | 作用 |
|---|---|---|
| Darknet-53 stage 3 | `[N,256,52,52]` | stride 8 的细粒度特征 |
| Darknet-53 stage 4 | `[N,512,26,26]` | stride 16 的中层特征 |
| Darknet-53 stage 5 | `[N,1024,13,13]` | stride 32 的深层特征 |
| 自顶向下融合 | stride 32 → 16 → 8 | 逐级上采样并拼接对应骨干特征 |
| 三个检测头 | `[N,S,S,3,5+C]`，S 为 13、26、52 | 每格三个 anchor；通道为 `tx,ty,tw,th,obj,class_logits` |

ConvNeXt-Small 变体从 TorchVision 的 stride 8、16、32 特征层取 `[N,192,52,52]`、`[N,384,26,26]`、`[N,768,13,13]`。neck/head 的连接方式和内部宽度不变，最初的 lateral 卷积输入通道随骨干特征通道调整。它按 ImageNet 权重使用 RGB mean/std 归一化；Darknet-53 路径仍使用 `[0,1]` 像素。TorchVision 文档将 `IMAGENET1K_V1` 标为 ImageNet-1K 分类权重，约 50.2M 参数；这不是 YOLO 检测预训练权重。[权重说明](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_small.html)

模型 `forward` 返回按 stride 32、16、8 排列的三个原始 logits 张量，不包含阈值筛选或 NMS。输入高宽须为 32 的倍数。主要实现位于 [model.py](../../x_yolo/models/yolov3/model.py)。

解码关系为：

```text
b_x = (sigmoid(tx) + cell_x) / grid_width
b_y = (sigmoid(ty) + cell_y) / grid_height
b_w = exp(tw) * anchor_w / input_width
b_h = exp(th) * anchor_h / input_height
score(class) = sigmoid(objectness) * sigmoid(class_logit)
```

## 标签、损失与后处理

每个真值框依据宽高 IoU，在全部九个 anchor 中选出最佳 anchor，并分到它所属的 stride 和中心所在网格。一个 cell/anchor 若遇到多个目标，保留 anchor 形状 IoU 较高者；其余计入 `ignored_ground_truths`，但仍会参与 objectness 的 IoU ignore 判断。

- `tx,ty` 使用带 `BCEWithLogits` 的 sigmoid 中心偏移；`tw,th` 回归相对 anchor 的对数宽高，采用平方误差。
- 正样本 objectness 与独立类别输出使用二元交叉熵；YOLOv2 的 softmax 类别和 IoU-rescore 目标不复用。
- 背景 anchor 与真值框 IoU 大于 `ignore_iou_threshold` 时不计入负 objectness 损失。默认阈值 `0.5` 取自官方 VOC cfg。
- 正样本坐标损失乘 Darknet 的 `(2 - box_area)` 权重。损失按 batch 大小归一化，并跨三个尺度相加。

论文将坐标训练目标概括为平方误差；Darknet `yolo_layer.c` 实际为中心偏移写入 `target - sigmoid(logit)` 的更新量。这里的 `BCEWithLogits` 对中心偏移产生相同的更新方向，但标量损失表达不同，因此文档分别记录论文描述与 Darknet 梯度实现，不把二者说成公式完全相同。
- 推理将三路预测合并，使用 `objectness × class_probability` 打分和 class-aware NMS，返回原图像素 `xyxy`、类别 ID 与置信度。

标签分配和损失位于 [loss.py](../../x_yolo/models/yolov3/loss.py)，解码与 NMS 位于 [postprocess.py](../../x_yolo/models/yolov3/postprocess.py)。

## 训练配置与运行

默认 recipe 是 `configs/yolov3/voc0712.yaml`：VOC 2007 train + VOC 2012 trainval 训练集、VOC 2007 val 验证集、VOC 2007 test 最终评估，416 输入、有效 batch 64、最多 200 epochs，每 10 epoch 保存可恢复的 `last.pt`，验证最优时保存 `best.pt`。初始方案采用显式 Darknet-53 分类预训练权重。固定 epoch 调度是项目训练器的工程设置，不等同于 Darknet cfg 的 `max_batches/steps/random=1` 调度。

单变量骨干对照 recipe 为 `configs/yolov3/voc0712_convnext_small.yaml`。它保持相同数据、anchors、neck/head 拓扑与内部宽度、标签规则、损失、优化器、输入尺寸、micro-batch 16、有效 batch 64、BF16 和训练 schedule，只把主干改为 ConvNeXt-Small；输入 lateral 卷积的通道会适配新主干。TorchVision 权重按 recipe 中明确写出的 `IMAGENET1K_V1` 加载；若本机缓存缺少权重，只有运行该训练 recipe 时才会下载。评估、推理构造模型时会直接从 checkpoint 恢复训练权重。

从官方页面取得 `darknet53.conv.74` 后，在被忽略的 `weights/` 目录保存：

```bash
mkdir -p weights/yolov3
curl -fL https://pjreddie.com/media/files/darknet53.conv.74 \
  -o weights/yolov3/darknet53.conv.74

python scripts/train_yolov3.py \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --pretrained-darknet53 weights/yolov3/darknet53.conv.74 \
  --device cuda \
  --output runs/yolov3/voc0712-darknet53
```

ConvNeXt-Small 对照运行时使用相同数据根目录和设备：

```bash
python scripts/train_yolov3.py \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --recipe configs/yolov3/voc0712_convnext_small.yaml \
  --device cuda \
  --output runs/yolov3/voc0712-convnext-small
```

权重加载器仅将文件中的 52 个卷积/BatchNorm 块装入 backbone；检测头随机初始化。模型不会静默下载权重。

评估和单图推理：

```bash
python scripts/evaluate_yolov3.py \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --checkpoint runs/yolov3/voc0712-darknet53/best.pt \
  --split val --device cuda

python scripts/infer_yolov3.py path/to/image.jpg \
  --checkpoint runs/yolov3/voc0712-darknet53/best.pt --device cuda

python scripts/evaluate_yolov3.py \
  --recipe configs/yolov3/voc0712_convnext_small.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --checkpoint runs/yolov3/voc0712-convnext-small/best.pt \
  --split val --device cuda
```

训练曲线、数据集统计和预测可视化沿用项目公共入口：

```bash
python scripts/plot_training_curves.py --run-dir runs/yolov3/voc0712-darknet53
python scripts/plot_dataset_analysis.py \
  --config configs/datasets/voc0712_trainmix.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --output-prefix runs/yolov3/voc0712-darknet53/dataset_analysis --grid-size 13
python scripts/visualize_yolov3_predictions.py \
  --recipe configs/yolov3/voc0712.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --checkpoint runs/yolov3/voc0712-darknet53/best.pt \
  --output-dir runs/yolov3/voc0712-darknet53/predictions --split val --device cuda
```

## 与 Darknet 参考实现的差异和验证状态

- 官方 VOC cfg 使用相同九个 anchor、三组 mask、每组 3 个框、`ignore_thresh=0.5`；这里保留这些检测约定。
- 本项目 dataset loader 将图像直接拉伸到方形。Darknet 的 VOC `valid` 路径也直接 resize；其单图 `test` 路径使用 letterbox。训练阶段不使用 letterbox，而是对图像与框做随机几何变换，并做 HSV 颜色扰动。
- recipe 固定 416 输入，不包含官方 cfg `random=1` 的随机多尺度训练；官方训练还用 `jitter=0.3`。本项目当前用固定输入、`scale_range=0.8–1.2`、`translate=0.2` 和 HSV 扰动，尚未加入随机水平翻转或同等的 jittered crop。训练 schedule 由 epoch、warmup 和衰减节点表达。
- 同一个 cell/anchor 的多目标冲突以“保留 anchor 形状 IoU 更高者”解决，是教学实现的确定性规则；Darknet 代码可能依赖目标遍历顺序处理碰撞。
- 当前训练 recipe 的几何增强是 scale/translation 与 HSV 颜色扰动，没有随机水平翻转。Darknet 与 ConvNeXt 两个 recipe 共用此数据流程，因此 backbone 对照不会把增强差异混入结果。
- ConvNeXt-Small + YOLOv3-head 变体已完成训练和 VOC 2007 test 评估：best checkpoint 为 epoch 185，test mAP@0.5 为 0.7937、mAP@0.5:0.95 为 0.4454。完整设置、学习率观察、逐类结果与可视化见[实验报告](../reports/yolov3_voc0712_convnext_small_20261008.md)。它是工程变体，不能将与 YOLOv1/v2 的差值单独归因于 YOLOv3 架构。
