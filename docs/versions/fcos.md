# FCOS

## 来源与定位

- 论文：Zhi Tian, Chunhua Shen, Hao Chen, Tong He, [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355), ICCV 2019。
- 参考实现：MMDetection 3.3.0 的 [FCOS ResNet-50/FPN 配置](https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py)，Apache-2.0。
- 本项目自行实现点分配、预测头、损失与后处理；TorchVision 仅提供 ResNet-50 和 FPN 特征模块，无 MMDetection 运行依赖。

## 核心思想

FCOS 不为每个位置枚举 anchor。每个 FPN 点直接预测类别、到真值框四边的距离 `l/t/r/b` 和 centerness。五个特征层分担不同大小的目标；centerness 降低远离目标中心的位置对最终分类分数的影响。

## 网络与张量

对 416×416 输入，ResNet-50 从 C3/C4/C5 特征构造 FPN 的 P3–P5，并向下扩展 P6/P7；对应 stride 是 8/16/32/64/128，空间尺寸是 52×52、26×26、13×13、7×7、4×4。每层共享一个分类塔和回归塔，塔内为 3×3 Conv + GroupNorm + ReLU。

每个金字塔层输出：

- `class_logits`: `[B, 20, H, W]`
- `box_distances`: `[B, 4, H, W]`，正距离以特征点为基准，乘以该层 stride 后得到输入画布像素距离
- `centerness_logits`: `[B, 1, H, W]`

解码时以 `(x+0.5, y+0.5) * stride` 作为点坐标，将 LTRB 转成画布 `xyxy`，按原图宽高分别缩放，再做 class-aware NMS。

## 标签分配与损失

点必须落在目标框内部，并符合对应层的最大边距范围：P3 `[0,64)`、P4 `[64,128)`、P5 `[128,256)`、P6 `[256,512)`、P7 `[512,+∞)` 像素。多个目标同时包含某个点时，选择面积较小的目标。第一版不启用 FCOSv2 的 center sampling。

- 分类：sigmoid Focal Loss，`alpha=0.25, gamma=2`
- 框：LTRB 解码后的 IoU loss，以 centerness target 加权
- 中心度：positive points 上的 sigmoid BCE

损失实现位于 [`x_yolo/models/fcos/loss.py`](../../x_yolo/models/fcos/loss.py)。点坐标、尺度范围、centerness 和预测解码留在 FCOS 自己的实现里，未混用 YOLO 的 anchor 或网格目标。

## 数据、权重与训练

数据使用 VOC2007 train + VOC2012 trainval 训练、VOC2007 val 选权重、VOC2007 test 做最终评估；类名顺序来自 [`configs/datasets/voc0712_trainmix.yaml`](../../configs/datasets/voc0712_trainmix.yaml)。目前数据 loader 把图像拉伸到固定正方形，便于共用同一数据合同；这和保持纵横比并 padding 的 MMDetection pipeline 不同。

预训练只初始化 ResNet-50 主干。配置采用 TorchVision `ResNet50_Weights.IMAGENET1K_V1`，与参考配置的 Caffe-pretrained ResNet 不同。训练脚本不会暗中下载权重；例如先准备文件：

```bash
curl -fL https://download.pytorch.org/models/resnet50-0676ba61.pth \
  -o /path/to/resnet50-0676ba61.pth
```

训练命令：

```bash
python scripts/train_dense_detector.py \
  --recipe configs/fcos/voc0712_resnet50.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --pretrained /path/to/resnet50-0676ba61.pth \
  --device cuda \
  --output runs/fcos/voc0712-resnet50
```

默认 recipe 使用 416 输入、120 epochs、SGD、前 3 epochs 冻结 ResNet、有效 batch 16、BF16，每 10 epochs 保存 checkpoint，每 5 epochs 验证。训练曲线由 `scripts/plot_training_curves.py` 生成，VOC 数据分布由 `scripts/plot_dataset_analysis.py` 生成。正式训练的结果、权重来源 hash 与测试图在对应报告和 `runs/fcos/` 目录中记录。

```bash
python scripts/evaluate_dense_detector.py \
  --recipe configs/fcos/voc0712_resnet50.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --checkpoint runs/fcos/voc0712-resnet50/best.pt \
  --split test --device cuda --output runs/fcos/voc0712-resnet50/test_metrics.json

python scripts/infer_dense_detector.py path/to/image.jpg \
  --recipe configs/fcos/voc0712_resnet50.yaml \
  --checkpoint runs/fcos/voc0712-resnet50/best.pt \
  --output-dir runs/fcos/voc0712-resnet50/inference --device cuda
```

## 与论文/参考实现的差异

- 主干使用 TorchVision ImageNet1K V1 ResNet-50，不是参考配置中的 Detectron Caffe-pretrained ResNet-50。
- 保留 P3–P7、LTRB、中心度、多层回归范围和 Focal/IoU/BCE 三类损失；训练不使用 COCO 1x 的 Caffe 归一化、proposal 配方或 Detectron 依赖。
- 固定输入尺寸和正方形拉伸，采用项目的几何/颜色增强，没有 Mosaic，也没有 FCOSv2 center sampling。
- 这是可追溯的 FCOS 教学实现，不声称逐项复现官方训练配方。

## 当前验证状态

CPU 小样例已检查 P3–P7 输出形状、正/空标注损失、反向梯度和空检测解码。完整 VOC 训练已完成 120 epochs，VOC 2007 test 上最佳验证 checkpoint 达到 mAP@0.5 `0.7777`、mAP@0.5:0.95 `0.5001`；训练环境、验证曲线、数据分析、逐类结果和可视化见[完整实验报告](../reports/fcos_voc0712_resnet50_20261009.md)。本地权重及运行产物不随源码提交。
