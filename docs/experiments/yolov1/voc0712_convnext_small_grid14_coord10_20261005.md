# YOLOv1 ConvNeXt Small：坐标损失权重对照
> 清理记录（2026-10-05）：旧 run 的 checkpoint、逐轮日志和图像产物已按请求清理；实验方案、主要指标和结论保留在本文，配方快照见[历史策略归档](historical_strategy_archive_20261005.md)。


## 问题与假设

当前 VOC2007+2012、ConvNeXt Small、14×14 方案的 mAP50 明显高于 mAP50:95。验证集检测上限敏感性实验显示，将每图上限从 100 提到 300 几乎不改变 AP，因此本实验转向训练目标：只把 YOLOv1 风格损失的坐标项权重 `lambda_coord` 从 5 提高到 10，观察严格定位指标是否改善。

这属于工程调参。原始 YOLOv1 论文坐标权重为 5；本实验不作为论文严格复现。

## 固定条件与改动

| 项目 | 当前最佳配方 | 本实验 |
| --- | --- | --- |
| 训练数据 | VOC2007 train + VOC2012 trainval | 相同 |
| 验证/测试 | VOC2007 val / test | 相同 |
| 模型 | ConvNeXt Small，448 输入，14×14 网格 | 相同 |
| 优化器和学习率 | Adam，head `1e-4`，backbone `1e-5`，epoch 13/19 衰减 | 相同 |
| 有效 batch、种子、增强 | 64、42、相同增强 | 相同 |
| 坐标损失权重 | 5 | **10** |

本实验从 ImageNet 预训练 backbone 初始化并从头训练检测头，没有从基线 checkpoint 接续。完整配置见 [`coord10 配方`](../../../configs/yolov1/voc0712_convnext_small_grid14_coord10.yaml)。

## 训练状态

运行目录：`runs/yolov1/voc0712-convnext-small-grid14-coord10-20261005`。计划 24 轮，验证间隔 2 轮，checkpoint 每 10 轮保存。训练期间观察到的验证结果：

| Epoch | `lambda_coord=10` mAP50 | `lambda_coord=10` mAP50:95 | 基线 `lambda_coord=5` mAP50 | 基线 mAP50:95 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.43053 | 0.12603 | 0.40682 | 0.11297 |
| 6 | 0.52669 | 0.17645 | 0.52257 | 0.16895 |
| 8 | 0.57199 | 0.20348 | 0.58538 | 0.21240 |
| 10 | 0.60132 | 0.23090 | 0.59729 | 0.22440 |
| 12 | 0.65195 | 0.26115 | 0.62116 | 0.24324 |
| 14 | 0.66760 | 0.28628 | 0.67280 | 0.28547 |
| 16 | 0.66964 | 0.28825 | 0.67322 | 0.28961 |
| 18 | 0.67440 | 0.29417 | 0.67793 | 0.29023 |
| 20 | 0.67711 | 0.29635 | 0.68223 | 0.29287 |
| 22 | 0.67680 | 0.29679 | 0.67846 | 0.29448 |
| 24 | **0.67712** | **0.29702** | 0.67924 | 0.29477 |

24 轮训练已完成，验证集最佳 checkpoint 是 epoch24。相对基线同 epoch，mAP50:95 增加 `0.00225`，mAP50 减少 `0.00212`。后段从 epoch20 到 24 的 mAP50:95 只增加 `0.00067`，学习率已降至 `1e-6`，模型接近当前配方的训练平台。

## VOC2007 test 对照

两个模型均使用 VOC2007 test 的 4,952 张图、相同预处理、置信度阈值、NMS 和每图 100 个预测框上限。`lambda_coord=10` 的 checkpoint 按验证集 mAP50:95 选择。

| 配方 | mAP50 | mAP50:95 | 检测数 |
| --- | ---: | ---: | ---: |
| `lambda_coord=5` | 0.67084 | 0.28683 | 491,997 |
| `lambda_coord=10` | 0.66679 | **0.28911** | 494,170 |
| 差值 | -0.00405 | **+0.00228** | +2,173 |

严格 IoU 汇总指标小幅提高，但 AP50 下降；20 类中 AP50:95 有 12 类提高，AP50 有 8 类提高。AP50:95 提升较多的类别包括 tvmonitor `+0.0219`、aeroplane `+0.0136`、bird `+0.0116` 和 dog `+0.0108`；下降较多的包括 motorbike `-0.0153`、bicycle `-0.0099` 和 sheep `-0.0097`。因此提高坐标权重呈现了“高 IoU 定位略好、宽松 IoU/AP50 有所回退”的权衡，收益有限且按类别不一致，不能把它当成彻底修复。

VOC2007 test 曾用于先前配方选择，不是全新盲测集；上述差值适合比较这两次项目实验，不代表独立 benchmark 结论。

## 框定位偏差诊断

为复现用户指出的现象，检查了基线模型固定 seed 42、阈值 0.25 的 20 张测试可视化样例（共 55 个 GT）。47 个 GT 有至少一个同类候选；对这 47 个候选匹配计算，中位 IoU 为 `0.683`，其中 `66%` 低于 IoU `0.75`。匹配框的中位 GT 覆盖率为 `0.857`，中位预测面积为 GT 的 `1.166` 倍，中位中心偏差为 GT 对角线的 `4.8%`。这组小样本更支持“框边缘/宽高偏松，中心整体偏移较小”，而不是统一的像素坐标偏移；它只用于定位问题，不代替全量测试指标。

检查了标签 target 编码、模型输出解码和原图像素缩放，并用合成框做 round-trip：标签经网格编码后再解码到像素坐标的最大误差为 `0 px`。没有发现绘图缩放或坐标轴转换 bug。实现里的定位项是中心偏移和 `sqrt(width/height)` 的均方误差；IoU 用来在两个预测框中选负责框，并作为置信度目标，但坐标项本身没有直接最小化 IoU。因此，**损失目标与最终 IoU 指标不一致是有证据支持的因素**，而不是已发现的损失实现错误。`lambda_coord=10` 在全量 test 上令 mAP50:95 提高 `0.00228`，也支持它对边界拟合有影响；但 AP50 同时下降，且整体增益很小，所以损失形式不是唯一原因。

另一个独立限制来自 YOLOv1 的目标分配：14×14 网格的每格只有一组类别分数，训练 target builder 对同格目标只保留最大框，其余不参与检测头监督。该训练集静态统计约有 616 个同格额外目标，随机几何增强后的每轮 `ignored_ground_truths` 约为 688–732。它更可能造成拥挤场景漏检或分类混淆。448 输入配 14×14 输出也只有 32 像素步长的特征网格，小目标的细边缘定位更难；固定样例中小目标只有 2 个，不能据此给总体小目标误差下结论。

## 产物

- 训练曲线 PNG（历史 run 产物已清理） / SVG（历史 run 产物已清理）
- VOC2007 test 逐类 AP 图（历史 run 产物已清理）
- 逐类 AP50:95 对照差值图（历史 run 产物已清理） / SVG（历史 run 产物已清理）
- 固定样例预测拼图（历史 run 产物已清理） 和逐图 manifest（历史 run 产物已清理）
- 固定样例定位统计及逐 GT 明细（历史 run 产物已清理）
- test 指标 JSON（历史 run 产物已清理） 与 相对 `lambda_coord=5` 的对照 JSON（历史 run 产物已清理）
- 数据集没有变化，沿用同一 VOC trainmix 数据集分析图（历史 run 产物已清理）。

下一轮若继续定位优化，应固定模型、数据、增强和优化器，只改变框回归目标，测试 IoU/GIoU 类定位项，并继续同时报告 AP50 与 AP50:95。当前证据表明这比继续盲目增加坐标权重更能直接检验“优化目标是否导致边界偏松”。

## 运行命令

```bash
python scripts/train_yolov1.py \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --recipe configs/yolov1/voc0712_convnext_small_grid14_coord10.yaml \
  --output runs/yolov1/voc0712-convnext-small-grid14-coord10-20261005 \
  --device cuda:0

python scripts/evaluate_yolov1.py \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --recipe configs/yolov1/voc0712_convnext_small_grid14_coord10.yaml \
  --checkpoint runs/yolov1/voc0712-convnext-small-grid14-coord10-20261005/best.pt \
  --split test --device cuda:0 --batch-size 4 \
  --output runs/yolov1/voc0712-convnext-small-grid14-coord10-20261005/test_metrics.json

python scripts/plot_training_curves.py \
  --run-dir runs/yolov1/voc0712-convnext-small-grid14-coord10-20261005
```
