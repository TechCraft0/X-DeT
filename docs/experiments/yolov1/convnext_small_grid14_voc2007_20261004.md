# ConvNeXt Small：7×7 与 14×14 网格受控对照
> 清理记录（2026-10-05）：旧 run 的 checkpoint、逐轮日志和图像产物已按请求清理；实验方案、主要指标和结论保留在本文，配方快照见[历史策略归档](historical_strategy_archive_20261005.md)。


## 问题与假设

YOLOv1 每个网格单元只分配一个真值类别与两个候选框。同一网格内多个物体中心会发生标签冲突。VOC2007 train 中 7×7 网格有 289 张冲突图像、421 个额外目标；本实验检查将输出网格改为 14×14 是否能减少冲突并改善检测。

14×14 改变了原论文的 7×7 设计，是本项目的工程变体，不是严格 YOLOv1 复现。为排除之前训练间 BF16 切换轮次不同造成的混淆，补跑了 7×7 对照组，使两组都在 epoch 1–10 使用 FP32、epoch 11–135 使用 BF16。

## 固定条件与唯一改动

两组均使用 TorchVision ConvNeXt Small `IMAGENET1K_V1` 预训练权重、VOC2007 自有 YOLO TXT 转换集、448×448 直接缩放、seed 42、Adam、检测头学习率 `1e-4`、backbone 学习率 `1e-5`、有效 batch 64（micro batch 4、累积 16 次）、冻结 backbone 5 轮后全量微调、epoch 75/105 学习率各衰减 10 倍、相同增强和损失权重，每 10 轮保存与验证。

唯一模型改动是检测网格：7×7 对照使用首层 stride 2，14×14 组使用 stride 1；标签分配、每格两个框、YOLOv1 损失、解码、NMS 及数据划分不变。两组各完成 135 个 epoch，`metrics.jsonl` 均记录 epoch 1–135；最佳权重按验证 mAP50:95 选择。

| 项目 | 7×7 匹配对照 | 14×14 |
| --- | --- | --- |
| 运行目录 | `runs/yolov1/voc2007-convnext-small-grid7-control-20261005` | `runs/yolov1/voc2007-convnext-small-grid14-20261004` |
| 验证最佳 mAP50:95 | epoch 110：0.25471 | epoch 110：0.26379 |
| 验证最佳 mAP50 | epoch 130：0.61231 | epoch 100：0.63725 |
| epoch 135 验证 mAP50 / mAP50:95 | 0.61073 / 0.25443 | 0.63453 / 0.26326 |

两组在 epoch 30 的验证指标近似相同；从 epoch 40 起 14×14 的 mAP50 逐渐领先。低学习率阶段两组的 mAP50:95 都继续提升到约 epoch 110，随后进入平台。7×7 从 epoch 110 到 135 没有明显提升，14×14 也仅小幅波动。训练总损失随 epoch 下降，没有发散证据；不同网格的目标和空网格数量不同，因此总损失绝对值不宜直接当作两种网格优劣指标。结果也不支持“只要继续增加 epoch 就会显著改善”。

训练环境记录为 NVIDIA GeForce RTX 4060 Ti 16 GB、PyTorch 2.10.0+cu128、TorchVision 0.25.0、CUDA 12.8、Python 3.10.12。两次运行均使用 ImageNet 预训练主干；该变体仍不同于 YOLOv1 原论文的大型全连接检测头。

## 运行命令

下列命令展示实际使用的两段精度日程。7×7 使用 `voc2007_convnext_small_convhead.yaml`，14×14 使用 `voc2007_convnext_small_grid14.yaml`；两组分别写入独立运行目录。

```bash
# 7×7：前 10 轮 FP32，再恢复并以 BF16 完成 135 轮
python scripts/train_yolov1.py \
  --dataset-root /path/to/yolo_voc2007 \
  --recipe configs/yolov1/voc2007_convnext_small_convhead.yaml \
  --output runs/yolov1/voc2007-convnext-small-grid7-control-20261005 \
  --device cuda:0 --epochs 10 --precision fp32
python scripts/train_yolov1.py \
  --dataset-root /path/to/yolo_voc2007 \
  --recipe configs/yolov1/voc2007_convnext_small_convhead.yaml \
  --output runs/yolov1/voc2007-convnext-small-grid7-control-20261005 \
  --device cuda:0 --precision bf16 \
  --resume runs/yolov1/voc2007-convnext-small-grid7-control-20261005/last.pt

# 14×14：相同精度日程和训练选项
python scripts/train_yolov1.py \
  --dataset-root /path/to/yolo_voc2007 \
  --recipe configs/yolov1/voc2007_convnext_small_grid14.yaml \
  --output runs/yolov1/voc2007-convnext-small-grid14-20261004 \
  --device cuda:0 --epochs 10 --precision fp32
python scripts/train_yolov1.py \
  --dataset-root /path/to/yolo_voc2007 \
  --recipe configs/yolov1/voc2007_convnext_small_grid14.yaml \
  --output runs/yolov1/voc2007-convnext-small-grid14-20261004 \
  --device cuda:0 --precision bf16 \
  --resume runs/yolov1/voc2007-convnext-small-grid14-20261004/last.pt
```

评估、曲线和数据集图使用项目内脚本生成；下面以 14×14 为例，7×7 将 recipe、checkpoint 和运行目录替换为对应项：

```bash
python scripts/evaluate_yolov1.py \
  --dataset-root /path/to/yolo_voc2007 \
  --recipe configs/yolov1/voc2007_convnext_small_grid14.yaml \
  --checkpoint runs/yolov1/voc2007-convnext-small-grid14-20261004/best.pt \
  --split test --device cuda:0 --batch-size 4 \
  --output runs/yolov1/voc2007-convnext-small-grid14-20261004/test_metrics.json
python scripts/plot_training_curves.py \
  --run-dir runs/yolov1/voc2007-convnext-small-grid14-20261004
python scripts/plot_dataset_analysis.py \
  --config configs/datasets/voc2007.yaml \
  --dataset-root /path/to/yolo_voc2007 \
  --output-prefix runs/yolov1/voc2007-convnext-small-grid14-20261004/dataset_analysis \
  --grid-size 14
python scripts/visualize_yolov1_predictions.py \
  --recipe configs/yolov1/voc2007_convnext_small_grid14.yaml \
  --dataset-root /path/to/yolo_voc2007 \
  --checkpoint runs/yolov1/voc2007-convnext-small-grid14-20261004/best.pt \
  --output-dir runs/yolov1/voc2007-convnext-small-grid14-20261004/predictions_test_conf025_matched \
  --split test --confidence 0.25 --nms-iou 0.45 --seed 42 --device cuda:0
```

## 网格碰撞分析

| Split | 图像数 | 网格 | 有中心冲突的图像 | 同格额外目标 |
| --- | ---: | ---: | ---: | ---: |
| train | 2,501 | 7×7 | 289（11.6%） | 421 |
| train | 2,501 | 14×14 | 78（3.1%） | 89 |
| val | 2,510 | 7×7 | 294（11.7%） | 501 |
| val | 2,510 | 14×14 | 98（3.9%） | 123 |
| test | 4,952 | 7×7 | 558（11.3%） | 823 |
| test | 4,952 | 14×14 | 156（3.1%） | 180 |

14×14 将 test 同格冲突图像减少约 72%，额外冲突目标减少约 78%。这与 mAP50 的提升方向相符，但碰撞数减少并不保证每类都提升。

## Test 结果

两组都用验证最佳 mAP50:95 checkpoint（epoch 110）在同一 VOC2007 test split 上评估，脚本参数为置信度下限 `0.001`、NMS IoU `0.45`、每图最多 100 个检测框。

| 模型 | Test mAP50 | Test mAP50:95 | 图像 | 输出检测数 |
| --- | ---: | ---: | ---: | ---: |
| ConvNeXt Small 7×7 | 0.62818 | 0.25931 | 4,952 | 489,725 |
| ConvNeXt Small 14×14 | **0.64769** | **0.26100** | 4,952 | 494,346 |
| 14×14 减 7×7 | **+0.01951** | **+0.00169** | — | — |

14×14 在 20 类中的 AP50 有 15 类上升，AP50:95 有 10 类上升。AP50:95 提升很小，说明它主要改善了检出/分类覆盖，整体高 IoU 定位质量几乎持平，不能据此宣称已经达到可部署水平。

变化较大的类别：

- AP50:95 上升：sheep `+0.0581`、cow `+0.0506`、bottle `+0.0345`、chair `+0.0327`、person `+0.0289`。
- AP50:95 下降：train `−0.0418`、diningtable `−0.0392`、cat `−0.0387`、bus `−0.0380`、aeroplane `−0.0197`。
- AP50 上 bottle、cow、sheep、person 增益较明显；diningtable、train、boat 有退步。

两组在置信度下限 `0.001` 时都接近每张图 100 个检测框上限（14×14 为平均 99.8 个/图）。这会截断低分候选，是当前 AP 评估的边界条件；`0.001` 是计算 AP 用的低分阈值，不是建议部署输出阈值。VOC 转换集省略了 `difficult=1` 标注，项目指标定义与官方 VOC devkit 结果不能直接横向对比。

test split 此前已用于预测图检查，所以这组结果可作为相同协议的受控 test 对照，但不应称为完全盲测。

## 推理置信度校准

只在 14×14 的 val split 上做阈值校准：先缓存低阈值预测，固定 NMS IoU `0.45` 和每图上限 100，再以步长 0.01 扫描置信度，统计 IoU 0.50 / 0.75 下的 macro、micro precision/recall/F1。此校准不改权重，也不改变上面的 AP 计算。

按 val macro F1@IoU0.50 选出的起始阈值是 `0.18`：macro precision `0.6844`、macro recall `0.6557`、macro F1 `0.6652`，micro F1 `0.6707`。阈值 `0.25` 时 macro F1 为 `0.6440`、precision `0.7586`、recall `0.5725`。因此 `0.18` 更偏向召回与类别均衡，`0.25` 更偏向精度；实际部署需按误检/漏检代价在 validation 上决定。IoU0.75 的最高 macro F1 只有 `0.3035`（阈值 0.21），再次显示定位仍是主要短板。

## 预测图检查

两组使用固定 seed 42 选择相同的 20 张 test 图像，统一以置信度 `0.25`、NMS `0.45` 并排可视化。大目标的坐标和缩放关系正常，没有发现整体坐标反变换错误；两组都仍会漏掉 chair、diningtable 等示例中的目标。瓶子示例标注 7 个实例，两组各输出 5 个框；pottedplant 示例标注 8 个实例，7×7 输出 3 个框、14×14 输出 2 个框。这个固定样例说明网格变密并没有解决所有拥挤目标，不能替代总体 AP 和召回统计。

## 产物

- 7×7/14×14 训练曲线对照（历史 run 产物已清理）
- 7×7：训练曲线 PNG（历史 run 产物已清理）、数据集分析图（历史 run 产物已清理）、逐类 Test AP（历史 run 产物已清理）、Test 指标 JSON（历史 run 产物已清理）、固定样例拼图（历史 run 产物已清理）。
- 14×14：训练曲线 PNG（历史 run 产物已清理）、数据集分析图（历史 run 产物已清理）、逐类 Test AP（历史 run 产物已清理）、Test 指标 JSON（历史 run 产物已清理）、同样例拼图（历史 run 产物已清理）、validation 置信度扫描曲线（历史 run 产物已清理）、阈值扫描 CSV（历史 run 产物已清理）。
- validation 选出阈值 0.18 后的 test 定性拼图：grid14 置信度 0.18（历史 run 产物已清理）。
- 最佳/最终 checkpoint、逐 epoch `metrics.jsonl`、配置快照与完整 JSON 报告都留在对应的 `runs/yolov1/` 目录。

## 判断与后续

控制了精度切换时间后，14×14 仍取得一致的验证 mAP 优势和 test AP50 增益，因此保留它作为当前较好的 ConvNeXt Small 候选。AP50:95 只增加 `0.00169`，类别收益分布不均，当前证据不足以称结果已经“可用”。训练曲线在约 epoch 110 后平台，单纯增加 epoch 不是优先方向；前面更高学习率续训试验也没有改善。

后续的 VOC2007+VOC2012 train 数据扩展已单独完成并记录在[曝光量匹配实验报告](voc0712_convnext_small_grid14_exposurematched_20261005.md)。该实验保留 VOC2007 val/test，VOC2007 test 结果仍受之前模型对比和可视化影响，不作为完全盲测；类别长尾和定位质量仍需单独研究。
