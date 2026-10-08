# CenterNet（Objects as Points）

## 来源与定位

- 论文：Kaiwen Duan et al., [Objects as Points](https://arxiv.org/abs/1904.07850), 2019。
- MMDetection 参考配置：[`centernet_r18_8xb16-crop512-140e_coco.py`](../../3dparty/mmdetection/configs/centernet/centernet_r18_8xb16-crop512-140e_coco.py)，本地参考 checkout 为 MMDetection `cfd5d3a985b0249de009b67d04f37263e11cdf3d`（Apache-2.0）。
- 公开检测权重：[CenterNet ResNet-18 COCO checkpoint](https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth)。checkpoint 是 COCO 检测权重，不是纯 ImageNet backbone。
- 本地下载文件大小 56,960,639 bytes，SHA256 `bb5b3bf7802b8b6e1777f925c2fd0948ba0328be87e30fd996ef1e06c86e6958`；文件保存在用户 cache，不提交到 Git。
- 本项目在 [`x_yolo/models/centernet/`](../../x_yolo/models/centernet/) 中实现独立模型、损失与解码，不依赖 MMDetection。

## 方法与张量流

CenterNet 把每个物体表示为其边界框中心点。网络在 stride-4 特征图上预测类别中心热图，并只在中心点位置回归宽高和亚像素 offset。这样不需要 anchor、候选框枚举或框 NMS。

首版使用 TorchVision ResNet-18 主干，接三层转置卷积 CTResNetNeck，把 stride-32 输出逐级上采样到 stride-4。512×512 输入得到 128×128 输出：

- `center_heatmap_logits`: `[B, 20, 128, 128]`
- `box_sizes`: `[B, 2, 128, 128]`，单位为输出特征格
- `center_offsets`: `[B, 2, 128, 128]`，每个维度位于一个格子的局部偏移

训练标签把真实框中心映射到特征图整数格，在类别热图上绘制 Gaussian；宽高与小数部分 offset 只对中心正样本计算 L1。总损失为 Gaussian Focal + `0.1 × L1(wh)` + `1.0 × L1(offset)`。推理保留 3×3 局部峰值，按分数取前 100 个类别中心，再读出宽高和 offset 解码回原图。

## 训练与评估

环境要求 Python 3.10+、PyTorch 与 TorchVision 版本匹配、Pillow、NumPy、PyYAML、Matplotlib。安装方式见仓库根目录 README。准备 VOC07+12 YOLO TXT 数据，类别顺序使用 [`voc0712_trainmix.yaml`](../../configs/datasets/voc0712_trainmix.yaml)。

先下载 COCO 检测 checkpoint；训练入口不会自动访问网络：

```bash
mkdir -p ~/.cache/x-det/checkpoints
curl -fL https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth \
  -o ~/.cache/x-det/checkpoints/centernet_resnet18_140e_coco.pth
```

完整训练 recipe：[`configs/centernet/voc0712_resnet18.yaml`](../../configs/centernet/voc0712_resnet18.yaml)。运行前可用 `--max-steps 2 --epochs 1` 检查保存流程；该路径不是收敛性或精度验证。

```bash
python scripts/train_keypoint_detector.py \
  --recipe configs/centernet/voc0712_resnet18.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --pretrained ~/.cache/x-det/checkpoints/centernet_resnet18_140e_coco.pth \
  --device cuda --output runs/centernet/voc0712-resnet18

python scripts/evaluate_keypoint_detector.py \
  --recipe configs/centernet/voc0712_resnet18.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --checkpoint runs/centernet/voc0712-resnet18/best.pt \
  --split test --device cuda \
  --output runs/centernet/voc0712-resnet18/test_metrics.json

python scripts/visualize_keypoint_predictions.py \
  --recipe configs/centernet/voc0712_resnet18.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --checkpoint runs/centernet/voc0712-resnet18/best.pt \
  --split test --device cuda \
  --output-dir runs/centernet/voc0712-resnet18/test_visualizations
```

也可通过 `scripts/infer_keypoint_detector.py IMAGE_OR_DIRECTORY` 对自有图片推理。每 10 epoch 保存 `last.pt`；validation `mAP50_95` 最优时保存 `best.pt`。曲线使用 `scripts/plot_training_curves.py`，数据分布图使用 `scripts/plot_dataset_analysis.py`。

## 实现差异与验证状态

- 保留经典 CenterNet ResNet-18、CTResNetNeck（DCNv2 关闭）、Gaussian center heatmap、WH/offset 回归和局部峰值解码；COCO 检测头按类名映射到 VOC 20 类。
- recipe 复用项目的数据加载器，固定拉伸到 512×512；上游使用 RandomCenterCropPad 和保持宽高比的 resize。增强和训练 schedule 按单卡 VOC 配置重设，结果不应称为严格复现 MMDetection COCO recipe。
- COCO checkpoint 可安全读取：168 个张量匹配，占 99.99996%；检测类别头按 VOC 类名映射。用真实 checkpoint 完成了 128×128 单 batch CPU 训练、验证和 best/last checkpoint 写入 smoke；它只验证链路，不代表精度。完整训练、VOC test 评估与可视化仍以最终实验报告为准。
- 暂未实现 CenterNet-Update。其 FPN、多尺度 head 和 GIoU 与这里的 Objects as Points 基线单独区分。
