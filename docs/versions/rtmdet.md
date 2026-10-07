# RTMDet

## 来源与定位

- 论文：Chengqi Lyu et al., [RTMDet: An Empirical Study of Designing Real-Time Object Detectors](https://arxiv.org/abs/2212.07784), 2022。
- 参考实现：MMDetection 3.3.0 的 [RTMDet-Tiny COCO 配置](https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py)，Apache-2.0。
- 本项目自行实现水平框版 Tiny 主干、颈部、检测头、动态软标签分配和解码，无 MMDetection 运行依赖。RTMDet-R2 的角度预测和旋转框算子不属于这个版本。

## 核心思想与张量流

RTMDet 组合 CSPNeXt、CSPNeXt PAFPN 与点距离检测头。Tiny 配置以 stride 8/16/32 的三个尺度输出特征；CSPNeXt 的 depthwise 5×5 卷积与通道注意力构成主干块，PAFPN 先自顶向下融合再自底向上聚合。

416×416 输入对应 P3/P4/P5 尺寸 52×52、26×26、13×13。Tiny 主干输出 96/192/384 通道，经 PAFPN 后各层为 96 通道。检测头在每一层使用分类塔和回归塔，卷积权重跨尺度共享，BN 按尺度分开：

- `class_logits`: `[B, 20, H, W]`
- `box_distances`: `[B, 4, H, W]`，四边距离已乘 stride，单位为输入画布像素

RTMDet-Tiny 配置采用线性回归距离（`exp_on_reg=False`）、无 objectness 分支和 point generator offset 0。推理把点位置与 LTRB 组合成 `xyxy`，还原原图尺度后做 class-aware NMS。

## 动态分配与损失

对位于任意真值框内部的候选点，计算预测框与真值框的 IoU、带软中心先验的距离代价和 Quality Focal 分类代价。每个 GT 根据 top-13 IoU 的和决定动态正样本数；冲突点分配给总代价最小的 GT。正样本的 IoU 同时作为分类软目标和损失权重。

- 分类：Quality Focal Loss，`beta=2`
- 框：按分配质量加权的 GIoU Loss，权重 2
- 匹配默认：top-k 13，soft center radius 3，IoU cost weight 3

实现位于 [`x_yolo/models/rtmdet/loss.py`](../../x_yolo/models/rtmdet/loss.py)。匹配过程只构造 target，不参与梯度传播。

## 预训练和训练

[`configs/rtmdet/voc0712_tiny.yaml`](../../configs/rtmdet/voc0712_tiny.yaml) 使用 OpenMMLab 发布的 RTMDet-Tiny COCO checkpoint。加载器以 PyTorch restricted unpickler 安全读取张量，兼容的主干、颈部和头部权重按名称与形状加载；三层 COCO 分类器按 COCO class name 映射出 Pascal VOC 的 20 个类别通道。权重不会自动下载或进入 Git。

```bash
curl -fL https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth \
  -o /path/to/rtmdet_tiny_coco.pth

python scripts/train_dense_detector.py \
  --recipe configs/rtmdet/voc0712_tiny.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --pretrained /path/to/rtmdet_tiny_coco.pth \
  --device cuda \
  --output runs/rtmdet/voc0712-tiny
```

recipe 使用 416 输入、100 epochs、AdamW、有效 batch 32、前 3 epochs 冻结 CSPNeXt、BF16；epoch 50 后余弦衰减，每 10 epochs 保存、每 5 epochs 验证。

```bash
python scripts/evaluate_dense_detector.py \
  --recipe configs/rtmdet/voc0712_tiny.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --checkpoint runs/rtmdet/voc0712-tiny/best.pt \
  --split test --device cuda --output runs/rtmdet/voc0712-tiny/test_metrics.json

python scripts/infer_dense_detector.py path/to/image.jpg \
  --recipe configs/rtmdet/voc0712_tiny.yaml \
  --checkpoint runs/rtmdet/voc0712-tiny/best.pt \
  --output-dir runs/rtmdet/voc0712-tiny/inference --device cuda
```

## 与论文/上游配方的差异

- 保留 Tiny CSPNeXt、PAFPN、分层 BN 头、动态软标签分配、QFL 和 GIoU 关键路径；实现命名对齐上游张量结构以便迁移公开预训练 checkpoint。
- 使用已在 COCO 检测上训练的 RTMDet-Tiny checkpoint，再对 VOC 类别继续微调；分类器按类名映射。输入从 640 改为 416。
- 沿用项目 YOLO TXT loader 的固定正方形缩放和轻量几何/颜色/水平翻转增强；没有实现缓存 Mosaic/MixUp、EMA 或最后阶段 pipeline switch。
- 优化器族为 AdamW，批量和学习率按单卡 VOC 规模调整，训练 100 epochs，而非上游 8 卡 COCO 300 epochs。
- 这是忠实保留主要结构与训练目标的独立工程实现，不是逐项复现 MMDetection 的完整数据流水线和分布式 recipe。

## 当前验证状态

CPU 小样例已检查三个尺度输出、动态分配、QFL/GIoU 反向梯度和解码；官方 RTMDet-Tiny EMA checkpoint 匹配 476 个张量，并完成 20 类分类通道迁移。完整 VOC 训练和 test 结果以后续实验报告为准。
