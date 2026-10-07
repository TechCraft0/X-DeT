# RTMDet-Tiny：VOC 训练与测试报告

## 结果

| checkpoint | 轮次 | VOC 2007 test 图像 | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| 验证集最佳 `best.pt` | 10 | 4,952 | **0.8171** | **0.6005** |
| 最终 `last.pt` | 100 | 4,952 | 0.8000 | 0.5966 |

验证集最佳权重的 VOC 2007 val 为 mAP@0.5 **0.8106**、mAP@0.5:0.95 **0.6014**。test 上 best 比 last 高 0.0171 / 0.0039；因此部署和发布采用验证集选出的 `best.pt`，不按 test 指标挑权重。每张图最多保留 100 个检测框，两个 test 评估均处理 4,952 张图、共 495,200 个输出框。

## 实验设置

- 数据：VOC 2007 train + VOC 2012 trainval 训练（14,041 张、33,751 框）；VOC 2007 val（2,510 张、6,307 框）选 checkpoint；VOC 2007 test（4,952 张、12,032 框）最终评估。类别分布与框尺寸见下方数据分析图。
- 模型：独立实现的 RTMDet-Tiny（CSPNeXt-Tiny、CSPNeXt-PAFPN、分层 BN 检测头、动态软标签分配、QFL 和 GIoU）。使用 OpenMMLab 发布的 COCO checkpoint，将 20 个 Pascal VOC 类别按类名从 COCO 分类器迁移。它保留 RTMDet 的核心结构和目标，不是 MMDetection 完整训练流水线的逐项复现。
- 输入与预处理：416×416；输入 RGB `[0,1]` 在模型内转为 RTMDet 配方所需 BGR 0–255，并按模型均值/标准差归一化。数据增强为水平翻转、轻量尺度/平移和颜色扰动。
- 训练：100 epochs；micro batch 8，梯度累积 4 次，有效 batch 32；BF16；前 3 epochs 冻结主干；AdamW，检测头 LR 0.0005、主干 LR 系数 0.25；epoch 50 后 cosine 衰减；每 10 epochs 保存、每 5 epochs 验证。100 轮日志中的纯训练时间合计 7,747 秒，未包含验证、启动和恢复时间。
- 初始化：`rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth`，SHA256 `78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb`。权重来自 OpenMMLab 的 MMDetection 3.3.0 RTMDet-Tiny COCO 发布文件。
- 环境：Python 3.10.12、PyTorch 2.10.0+cu128、TorchVision 0.25.0+cu128、CUDA 12.8、Pillow 11.2.1、NumPy 1.26.4；NVIDIA GeForce RTX 4060 Ti 16GB，BF16 可用。

运行配置保存了完整 recipe、数据集划分和设备信息：[`run_config.yaml`](../../runs/rtmdet/voc0712-tiny-coco-init-416-bf16-bs32-20261008/run_config.yaml)。训练从 epoch 40 检查点恢复一次，以启用已验证等价的批量 AP 汇总实现；模型参数、优化器状态、训练配方和 100 epoch 总数均连续保留。

## 曲线观察与权重选择

| Epoch | 训练总损失 | 验证 mAP@0.5 | 验证 mAP@0.5:0.95 | 检测头 LR |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 0.623 | 0.8100 | 0.5996 | 0.000500 |
| **10** | 0.543 | 0.8106 | **0.6014** | 0.000500 |
| 25 | 0.449 | **0.8140** | 0.5993 | 0.000500 |
| 40 | 0.404 | 0.8025 | 0.5840 | 0.000500 |
| 60 | 0.358 | 0.7924 | 0.5733 | 0.000455 |
| 80 | 0.291 | 0.7952 | 0.5881 | 0.000189 |
| 95 | 0.265 | 0.7964 | 0.5928 | 0.0000366 |
| 100 | 0.261 | 0.7948 | 0.5896 | 0.0000250 |

训练损失从 0.713 降到 0.261；验证 mAP@0.5 大多在 0.79–0.81，定位更敏感的 mAP@0.5:0.95 在前 10 轮达到最佳，epoch 40–65 下滑，学习率衰减后回到约 0.59。best 和 last 的 test 差距不大，但 best 仍领先，说明主要风险是中期泛化/定位波动，后期衰减只能部分恢复。recipe 未实现上游的 Mosaic、MixUp、EMA 和末段数据流水线切换；在此 VOC 设置中，后续可单独验证这些增强对拥挤、小目标定位的影响。

## 与 YOLO 结果的参考对照

| 模型与权重 | VOC 2007 test mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: |
| YOLOv1 ConvNeXt-Small 工程版 | 0.6875 | 0.3185 |
| YOLOv2 项目实现 | 0.6936 | 0.3941 |
| YOLOv3-head + ConvNeXt-Small 工程版 | 0.7937 | 0.4454 |
| RTMDet-Tiny，validation best | **0.8171** | **0.6005** |

这些结果使用同一 VOC 2007 test split，但主干预训练来源、网络容量、输入、损失、增强和训练配方不同。它们适合项目内了解现有实现的实测表现，不能当作只改变检测器结构的严格受控消融。当前 test AP 显示 RTMDet-Tiny 这套 COCO 预训练和 VOC 微调设置更强，尤其在较严格 IoU 下；后续若要比较结构本身，需要进一步固定主干初始化、数据增强和训练预算。

逐类 test AP@0.5:0.95 的强项包括 bus 0.772、car 0.721、cat 0.710；较弱类别为 pottedplant 0.302、chair 0.410、boat 0.474、bottle 0.479。测试可视化中大目标通常框得较准，椅子、植物和拥挤场景有漏检或跨类别误检；固定展示阈值为 0.25，类别 AP 使用低分候选参与排序，因此这两类证据回答的问题不同。

## 可视化与产物

运行目录：`runs/rtmdet/voc0712-tiny-coco-init-416-bf16-bs32-20261008/`（本地实验数据与 checkpoint 不纳入 Git 源码提交）。

- [`training_curves.png`](../../runs/rtmdet/voc0712-tiny-coco-init-416-bf16-bs32-20261008/training_curves.png) / SVG：训练损失、逐类验证 AP、学习率和正样本数。
- [`dataset_analysis.png`](../../runs/rtmdet/voc0712-tiny-coco-init-416-bf16-bs32-20261008/dataset_analysis.png) / JSON：VOC 各 split 类别数量、框尺寸和网格分布。
- [`test_best.json`](../../runs/rtmdet/voc0712-tiny-coco-init-416-bf16-bs32-20261008/test_best.json) 与 [`test_last.json`](../../runs/rtmdet/voc0712-tiny-coco-init-416-bf16-bs32-20261008/test_last.json)：测试整体及逐类 AP。
- [`best checkpoint 可视化`](../../runs/rtmdet/voc0712-tiny-coco-init-416-bf16-bs32-20261008/visualizations_best/contact_sheet.jpg) / [`manifest`](../../runs/rtmdet/voc0712-tiny-coco-init-416-bf16-bs32-20261008/visualizations_best/manifest.json)：固定种子从 test 每类抽取一张图，展示标注与预测。
- [`last checkpoint 可视化`](../../runs/rtmdet/voc0712-tiny-coco-init-416-bf16-bs32-20261008/visualizations_last/contact_sheet.jpg)：相同样本，便于观察训练末期差异。
- [`单图推理结果`](../../runs/rtmdet/voc0712-tiny-coco-init-416-bf16-bs32-20261008/example_dog_inference/007181_prediction.jpg)：validation best checkpoint，confidence 0.25。

![RTMDet training curves](../../runs/rtmdet/voc0712-tiny-coco-init-416-bf16-bs32-20261008/training_curves.png)

![RTMDet VOC test predictions](../../runs/rtmdet/voc0712-tiny-coco-init-416-bf16-bs32-20261008/visualizations_best/contact_sheet.jpg)

## 复现和推理

训练需要本地 VOC YOLO-TXT 转换数据与显式 RTMDet-Tiny 预训练权重；训练脚本不会隐式下载。命令、转换要求和权重来源见[RTMDet 版本说明](../versions/rtmdet.md)。使用训练出的权重：

```bash
python scripts/infer_dense_detector.py path/to/image.jpg \
  --recipe configs/rtmdet/voc0712_tiny.yaml \
  --checkpoint runs/rtmdet/voc0712-tiny-coco-init-416-bf16-bs32-20261008/best.pt \
  --output-dir runs/rtmdet/inference --device cuda --confidence 0.25
```
