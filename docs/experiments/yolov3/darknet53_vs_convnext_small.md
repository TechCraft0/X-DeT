# YOLOv3：Darknet-53 与 ConvNeXt-Small 主干对照

## 问题与假设

YOLOv3 的 VOC 结果是否受主干特征表达限制？用现代 TorchVision ConvNeXt-Small ImageNet-1K 预训练权重替换 Darknet-53，其他训练条件尽量固定，观察 mAP50 与 mAP50:95 是否一起提升。

当前 YOLOv3 基线并非随机初始化主干：Darknet 官方训练页说明 `darknet53.conv.74` 是 ImageNet 分类预训练卷积权重。ConvNeXt-Small 同样使用 ImageNet-1K 分类预训练；这项实验比较的是骨干结构和其对应预训练，而不是“随机骨干 vs 预训练骨干”。[Darknet 预训练说明](https://pjreddie.com/darknet/yolo/)

## 对照设置

| 条件 | Darknet-53 基线 | ConvNeXt-Small 变体 |
|---|---|---|
| 训练数据 | VOC2007 train + VOC2012 trainval（14,041 张） | 相同 |
| 验证 / 最终测试 | VOC2007 val（2,510 张）/ VOC2007 test（4,952 张） | 相同 |
| 输入与 anchors | 416×416，官方 VOC 九个 anchors | 相同 |
| 检测结构 | YOLOv3 三尺度 neck/head、内部宽度和损失 | 拓扑与内部宽度相同，lateral 输入通道按 ConvNeXt 特征调整 |
| 增强 | scale/translation + HSV；固定尺寸，无随机水平翻转 | 相同 |
| 优化 | SGD，初始 LR 0.001，momentum 0.9，weight decay 0.0005 | 相同 |
| 训练预算 | 200 epochs，micro-batch 16，effective batch 64，BF16，seed 42 | 相同 |
| 主干初始化 | Darknet-53 ImageNet 分类预训练 | TorchVision `IMAGENET1K_V1` |

代码使用 YOLOv2 和 YOLOv3 共用的 AP 评估实现，VOC07 test 均为 4,952 张图，置信度阈值 `0.001`、class-aware NMS IoU `0.45`、每图最多 100 个框。训练集、验证集和测试集没有交叉；模型按验证集 mAP50:95 选优，再对 VOC07 test 作一次最终评估。

ConvNeXt 变体是 YOLOv3 检测头工程实验，不是论文原版 YOLOv3。模型按各自主干正确执行输入归一化：Darknet 使用 `[0,1]` RGB，ConvNeXt 使用 TorchVision ImageNet mean/std。

## 当前证据

| 运行 | 验证集 mAP50 | 验证集 mAP50:95 | VOC test / 试验状态 |
|---|---:|---:|---|
| YOLOv3 Darknet-53，epoch 190/200 最佳 | 0.71875 | 0.43046 | **0.71605 / 0.42946** |
| YOLOv3 ConvNeXt-Small，无冻结试跑 | epoch 5：0.000009 | epoch 5：0.000002 | 未评估 |
| YOLOv3 ConvNeXt-Small，冻结 10 轮，解冻 LR=0.001 | epoch 10：0.58165 | epoch 10：0.23545 | 已停止；epoch 15 val：0.000179 / 0.000033 |
| YOLOv3 ConvNeXt-Small，冻结 10 轮，解冻 LR=0.0001 | epoch 25：0.70958 | epoch 25：0.32269 | 训练中 |
| YOLOv2 Darknet-19，epoch 165 最佳 | — | — | 0.69365 / 0.39408 |

Darknet-53 YOLOv3 在相同 VOC07 test 上比 YOLOv2 高 0.02241 mAP50、0.03538 mAP50:95（分别约 2.24 和 3.54 个百分点）。这是当前实现和训练配方的整体对比，不能将差异只归因于 YOLOv3 的网络结构。ConvNeXt-Small 的训练正在进行，结束后按相同流程测试。

Darknet-53 基线已训练满 200 epochs，最佳 checkpoint 选在 epoch 190；最终 epoch 200 的 validation 为 0.71772 / 0.42882。训练曲线、VOC07 test 指标和按类别可视化保存在 `runs/yolov3/voc0712-darknet53-416-bf16-bs16-20261006/`。目视检查的样例在常见单目标类别上框与物体贴合；多人、多物体、遮挡及小物体类别仍可见错检或定位松紧差异，尤其 bottle、chair、pottedplant 的 AP50:95 较低，后续不应只看汇总 mAP。

ConvNeXt 的首个全量微调试跑在 epoch 10 停止：epoch 5 val mAP50 / mAP50:95 为 `0.000009 / 0.000002`，epoch 10 已无超过置信度阈值的检测。抽查 epoch 5 的 `000005.jpg` 预测，最高 chair 分数约 `0.0014`，框未覆盖标注物体。loss 从 epoch 1 的 171 降至 epoch 10 的 31.51，但 mAP 没有随之改善，因此不继续浪费该配方的训练预算。试跑权重与日志保存在 `runs/yolov3/voc0712-convnext-small-416-bf16-bs16-20261007/`，其中 `best.pt` 是 epoch 5、`last.pt` 是 epoch 10。

冻结对照在 epoch 5 已恢复到有实际意义的检测表现：mAP50 `0.45243`、mAP50:95 `0.16339`；同一轮全量微调几乎为零。第 10 轮冻结结束时达到 `0.58165 / 0.23545`，Darknet-53 同期为 `0.62898 / 0.26199`。解冻后若仍用 LR `0.001`，第 15 轮跌至 `0.000179 / 0.000033`，训练 loss 也从冻结末期 `17.01` 上升到 `31.78`。因此已停止这条路线并保留日志；证据表明主要问题是解冻时 backbone 更新幅度过大。

新对照从冻结阶段 epoch 10 checkpoint 恢复，只把 backbone LR 设为 detector LR 的 `0.1` 倍，也就是基础 LR `0.001` 下 backbone LR `0.0001`；解冻阶段其余条件一致。恢复代码现会保留当前 recipe 中设置的 LR multiplier，而不会被旧 checkpoint 的 optimizer group 元数据覆盖。

低 LR 解冻路线第 15 轮达到 val mAP50 `0.66186`、mAP50:95 `0.28764`，相对 epoch 10 冻结结果分别提高 `0.08021 / 0.05219`；也超过 Darknet-53 第 15 轮的 `0.62301 / 0.27888`。第 20 轮 ConvNeXt 为 `0.68714 / 0.30239`，Darknet-53 为 `0.67385 / 0.31299`：ConvNeXt 的 mAP50 高 `0.01329`，mAP50:95 低 `0.01061`。到第 25 轮，ConvNeXt 达到 `0.70958 / 0.32269`，Darknet-53 为 `0.66872 / 0.30218`，两项分别高 `0.04087 / 0.02051`。相比同样 epoch 15、主干仍用 `0.001` 更新的路线（`0.000179 / 0.000033`），这支持“预训练主干解冻时需要更小学习率”的判断。这里只能得出验证集上的中期结论，最终以各自 val 最佳 checkpoint 的同一 VOC07 test 对照为准。

VOC07 test 的 YOLOv2 / YOLOv3 按类别对照图为 `runs/yolov3/voc0712-darknet53-416-bf16-bs16-20261006/yolov2_comparison.png`，明细 CSV 同目录。YOLOv3 在 20 类中的 15 类 AP50:95 更高；较明显的提升出现在 sheep、aeroplane、bottle 和 bus，pottedplant、train 等类别仍低于 YOLOv2。

截至 epoch 150，mAP50:95 在 0.38 左右波动，没有刷新 epoch 130 的最佳 0.39117。学习率于 epoch 160 降到 0.0001 后，epoch 165/170/175/180/190 的 mAP50 与 mAP50:95 依次为 0.70943/0.41942、0.71504/0.42379、0.71630/0.42637、0.71936/0.42945、0.71875/0.43046；相较 epoch 160 的 0.68803/0.38468，epoch 190 的 AP50:95 又提高到 0.43046。当前证据支持学习率衰减有助于定位细化，epoch 180 和 190 则体现宽松 IoU 与严格 IoU 指标并非同步变化；仍要观察最终 test。

代码验证方面，YOLOv3 定向测试为 8 passed，随后完整仓库测试为 27 passed；TorchVision 特征权重以 `strict=True` 成功加载，并确认首层卷积权重与官方模型相同；64×64 输入产生 stride 32/16/8 三路预期形状。ConvNeXt-Small batch 16、BF16 的 forward/backward 显存预检峰值为 5.9 GiB。前 10 轮冻结对照已启动；预检只验证训练路径和显存，不代表训练收敛或性能结论。

## 可复现配置

- Darknet-53：[configs/yolov3/voc0712.yaml](../../../configs/yolov3/voc0712.yaml)
- ConvNeXt-Small：[configs/yolov3/voc0712_convnext_small.yaml](../../../configs/yolov3/voc0712_convnext_small.yaml)
- ConvNeXt-Small（前 10 epoch 冻结）：[configs/yolov3/voc0712_convnext_small_freeze10.yaml](../../../configs/yolov3/voc0712_convnext_small_freeze10.yaml)
- ConvNeXt-Small（前 10 epoch 冻结，backbone LR 0.1x）：[configs/yolov3/voc0712_convnext_small_freeze10_backbone_lr01.yaml](../../../configs/yolov3/voc0712_convnext_small_freeze10_backbone_lr01.yaml)
- 版本说明：[docs/versions/yolov3.md](../../versions/yolov3.md)
- 论文：[YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
- ConvNeXt 权重定义：[TorchVision ConvNeXt-Small](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_small.html)

正式结论待 ConvNeXt 训练、两份 VOC test 指标、曲线与预测可视化完成后更新。
