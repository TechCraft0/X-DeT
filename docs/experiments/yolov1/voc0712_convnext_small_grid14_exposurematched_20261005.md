# ConvNeXt Small YOLOv1 风格检测器：扩充 VOC 训练数据
> 清理记录（2026-10-05）：旧 run 的 checkpoint、逐轮日志和图像产物已按请求清理；实验方案、主要指标和结论保留在本文，配方快照见[历史策略归档](historical_strategy_archive_20261005.md)。


## 结论

在保持 14×14 检测网格、ConvNeXt Small 主干和 YOLOv1 风格损失不变的情况下，将训练数据由 VOC2007 train 扩展为 VOC2007 train + VOC2012 trainval。总训练图像曝光量与原 VOC2007 配方相差约 0.16%。

混合集实验在 VOC2007 test 上取得 mAP50 **0.67084**、mAP50:95 **0.28683**。对照同一脚本、同一 VOC2007 test split 和相同后处理设置的 VOC2007-only grid14 模型，分别提升 **0.02315** 和 **0.02583**；20 类中有 15 类 AP50、17 类 AP50:95 上升。数据扩充是目前最有证据支持的改进。

这仍是使用 TorchVision 预训练主干和 YOLOv1 风格卷积检测头的教学/工程变体，不是原论文严格复现。AP 评估每图最多保留 100 个候选，实际输出平均约 99.4 个/图；VOC 转换时排除了 `difficult=1` 目标。VOC2007 test 也已用于之前的模型对比和可视化，因此结果不是盲测。

## 问题与实验设计

上一轮 ConvNeXt Small 14×14 模型在 VOC2007 test 上为 0.64769 / 0.26100（mAP50 / mAP50:95）。原 VOC2007 train 只有 2,501 张图，类别数量差异明显。本实验假设是：在不增加总训练曝光量的情况下，加入 VOC2012 图像可改善数据多样性与类别覆盖。

主要改变是训练图像来源。为使每图重复训练次数接近，VOC2007-only 配方训练 135 轮，混合集训练 24 轮；训练 batch、优化器、主干、检测头、损失和增强保持相同。学习率节点、冻结时长和 FP32 阶段按总曝光量比例缩放。因 epoch 边界的梯度累积尾批处理，两组优化器更新总数仍相差约 2.2%，这是本对比的残余差异。

| 项目 | VOC2007-only grid14 | VOC2007 + VOC2012 grid14 |
| --- | ---: | ---: |
| train 图像 | 2,501 | 14,041 |
| train loader 每轮实际使用图像 | 2,500 | 14,040 |
| 训练轮数 | 135 | 24 |
| 总训练图像曝光 | 337,500 | 336,960 |
| 总 micro-batches（metrics 的 `global_step`） | 84,375 | 84,240 |
| 有效 batch / micro batch | 64 / 4 | 64 / 4 |
| 学习率衰减轮次 | 75、105 | 13、19 |
| 冻结主干轮数 | 5 | 1 |
| FP32 阶段 | 前 10 轮 | 前 2 轮 |

`global_step` 在本项目中计数的是 micro-batch，不是 optimizer update。每 16 个 micro-batch 累积后更新，epoch 结束时也会处理剩余梯度。因此两组约为 5,400 次和 5,280 次 optimizer update。冻结与 FP32 阶段的绝对图像数接近，但不完全相同。

## VOC 数据来源与转换

VOC2012 trainval 取自 [PASCAL VOC 官方页面](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/)。下载归档大小为 1,999,639,040 字节，MD5 为 `6cd6e144f989b92b3379bac3b3de84fd`，已在提取前校验。原始归档和既有 VOC2007 数据目录保持不变；YOLO TXT 标注写入独立转换目录，混合集再复制到独立派生数据集。

| Split | 图像 | 框 | 14×14 同格冲突图像 | 同格额外目标 |
| --- | ---: | ---: | ---: | ---: |
| train（VOC07 train + VOC12 trainval） | 14,041 | 33,751 | 506（3.6%） | 616 |
| val（VOC07 val） | 2,510 | 6,307 | 98（3.9%） | 123 |
| test（VOC07 test） | 4,952 | 12,032 | 156（3.1%） | 180 |

VOC2012 trainval 转换后为 11,540 张图、27,450 个框；`difficult=1` 的 4,111 个对象未写入普通 YOLO TXT，未识别类别为 0。对 VOC2012 trainval 与 VOC2007 train/val/test 做逐文件 SHA256 检查，未发现完全相同的图像。类别顺序沿用项目 VOC20 类配置。

混合集 train 的 `person` 占 10,924 / 33,751 个框（32.4%），`sofa` 最少，为 690 个框（2.0%）。长尾仍存在，但相较 VOC2007-only，最少见类别的绝对样本数已显著增加。

## 模型与训练配方

- 输入 448×448，输出网格 14×14，每格 2 个候选框，20 个 VOC 类别。
- TorchVision ConvNeXt Small `IMAGENET1K_V1` 预训练权重，后接项目的 YOLOv1 风格卷积检测头。
- Adam；检测头初始学习率 `1e-4`，backbone 学习率为其 0.1 倍；权重衰减 0。
- 有效 batch 64（micro batch 4、梯度累积 16），seed 42。
- 前 1 轮冻结 backbone；第 2 轮开始全量训练。前 2 轮 FP32，后续 BF16。
- 学习率在第 13、19 轮衰减 10 倍；验证每 2 轮运行一次；每 10 轮保存训练 checkpoint。
- 颜色、缩放和平移增强沿用基线配方。

实际训练 24/24 轮完成。训练总 loss 从第 1 轮的 7.127 降至第 24 轮的 2.327。最佳验证 checkpoint 为 epoch 24：mAP50 `0.67924`、mAP50:95 `0.29477`。在相同累计图像曝光量附近，VOC2007-only run 的末轮验证为 `0.63453 / 0.26326`；混合集结果高约 `0.04471 / 0.03151`。旧 run 验证间隔更稀，且两组的 optimizer update 数略有差异，因此该曲线对照用于判断趋势，不视为严格同一步数重复实验。

## VOC2007 test 结果

两组使用各自验证集选出的权重，评估相同 VOC2007 test split。置信度下限 `0.001`、NMS IoU `0.45`、每图最多 100 个检测；指标为项目内 continuous integral AP，不等同于 VOC 官方 devkit 指标。

| 模型 | mAP50 | mAP50:95 | 图像 | 预测框 |
| --- | ---: | ---: | ---: | ---: |
| ConvNeXt Small 7×7 / VOC07 | 0.62818 | 0.25931 | 4,952 | 489,725 |
| ConvNeXt Small 14×14 / VOC07 | 0.64769 | 0.26100 | 4,952 | 494,346 |
| ConvNeXt Small 14×14 / VOC07+12 | **0.67084** | **0.28683** | 4,952 | 491,997 |
| 混合集减 VOC07-only grid14 | **+0.02315** | **+0.02583** | — | — |

混合集相对 VOC07-only grid14 的 AP50 在 15/20 类上升，AP50:95 在 17/20 类上升。AP50:95 增益较大的类别为 dog `+0.0690`、sheep `+0.0629`、cat `+0.0614`、tvmonitor `+0.0544`、cow `+0.0381`。horse 的 AP50:95 下降 `0.0290`；car 下降 `0.0035`，sofa 下降 `0.0007`。

新模型共输出 491,997 个框，平均 99.35 个/图，接近 100 框上限。这提示低置信候选可能被截断，绝对 AP 会受到评估候选上限影响；但两组对照采用相同上限，所以上表的数据规模比较仍保持一致。下一步若研究 cap，应先在 validation 上对比不同 `max_detections`，再固定设置重跑对照。

## 每图检测上限敏感性

为了检查 100 框上限是否截断了有效召回，固定 epoch 24 checkpoint、VOC2007 val、置信度阈值 `0.001` 和 NMS IoU `0.45`，只改变每图保留框数。模型前向只运行一次，然后对同一组 NMS 排序结果分别按 100、200、300 截断计算 AP。

| 每图上限 | mAP50 | mAP50:95 | val 总预测框 |
| ---: | ---: | ---: | ---: |
| 100 | 0.67931 | 0.29474 | 249,540 |
| 200 | 0.67939 | 0.29476 | 468,637 |
| 300 | 0.67939 | 0.29477 | 616,206 |

300 框比 100 框多输出 366,666 个检测，但 mAP50 只增 `0.00008`、mAP50:95 只增 `0.00003`。因此每图 100 框不是当前验证 AP 的主要瓶颈，继续增大上限只会扩大输出。该敏感性重算使用 batch size 8；训练期间验证使用 batch size 4，两者的 cap=100 指标差小于 `0.0001`。

- 检测上限敏感性曲线 PNG（历史 run 产物已清理） / SVG（历史 run 产物已清理）
- 逐档指标 JSON（历史 run 产物已清理）

## 曲线与预测检查

训练曲线显示，验证 mAP50 与 mAP50:95 均随训练提升，14 轮后仍有小幅改善，24 轮是 mAP50:95 最佳点。第 13、19 轮学习率衰减后没有验证性能崩落。训练 loss 继续缓慢下降，后段验证增益小于前段，说明收益趋缓但尚未出现明显反向过拟合。

固定 seed 42、置信度 `0.25` 的 20 类样例拼图与 VOC2007-only run 使用同样的样例选择规则。普通目标上没有观察到整体坐标映射偏移；拥挤的 bottle/car、pottedplant 和 diningtable 样例仍会漏检或混类。拼图是定性检查，不代表总体召回率。

## 产物与复现命令

- 训练曲线 PNG（历史 run 产物已清理） / SVG（历史 run 产物已清理）
- 数据集分析 PNG（历史 run 产物已清理） / SVG（历史 run 产物已清理）
- VOC2007 test 模型对比图（历史 run 产物已清理）
- 按训练曝光量对齐的验证曲线（历史 run 产物已清理）
- 逐类 AP50/AP50:95（历史 run 产物已清理），逐类 AP50:95 差值（历史 run 产物已清理）
- test 固定样例拼图（历史 run 产物已清理） 与逐图 `manifest.json`
- `best.pt`、`last.pt`、`metrics.jsonl`、dataset/run 配置快照及 test_metrics.json（历史 run 产物已清理）

可复用入口：

```bash
# 把 VOC_DATA_ROOT 指向包含派生 YOLO TXT 数据集的本地 VOC 数据根目录。
python scripts/train_yolov1.py \
  --dataset-root "$VOC_DATA_ROOT/yolo_voc2007_2012_trainmix" \
  --recipe configs/yolov1/voc0712_convnext_small_grid14_exposurematched.yaml \
  --output runs/yolov1/voc0712-convnext-small-grid14-exposurematched-20261005 \
  --device cuda:0 --epochs 2 --precision fp32
python scripts/train_yolov1.py \
  --dataset-root "$VOC_DATA_ROOT/yolo_voc2007_2012_trainmix" \
  --recipe configs/yolov1/voc0712_convnext_small_grid14_exposurematched.yaml \
  --output runs/yolov1/voc0712-convnext-small-grid14-exposurematched-20261005 \
  --device cuda:0 --precision bf16 \
  --resume runs/yolov1/voc0712-convnext-small-grid14-exposurematched-20261005/last.pt
python scripts/evaluate_yolov1.py \
  --dataset-root "$VOC_DATA_ROOT/yolo_voc2007_2012_trainmix" \
  --recipe configs/yolov1/voc0712_convnext_small_grid14_exposurematched.yaml \
  --checkpoint runs/yolov1/voc0712-convnext-small-grid14-exposurematched-20261005/best.pt \
  --split test --device cuda:0 --batch-size 4 \
  --output runs/yolov1/voc0712-convnext-small-grid14-exposurematched-20261005/test_metrics.json
python scripts/plot_training_curves.py \
  --run-dir runs/yolov1/voc0712-convnext-small-grid14-exposurematched-20261005
```

## 下一步

保留 VOC2007+2012、14×14、ConvNeXt Small 为当前最强候选。AP50 与 AP50:95 同时提升，且多数类别受益，说明此前训练样本量和类别覆盖不足确实限制了结果。检测上限敏感性已排除 `max_detections=100` 是主要瓶颈。AP50:95 与 AP50 之间仍有较大差距，下一轮固定数据、模型和训练配方，只将 YOLOv1 坐标损失权重从 `lambda_coord=5` 提到 `10`，观察高 IoU AP 是否改善；该变化是工程调参，不再等同原论文权重设置。
