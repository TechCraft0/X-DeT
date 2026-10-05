# VOC2007 推理可视化问题诊断

日期：2026-10-01。对象是[修正增强后完整训练](voc2007_yolov1_resnet50_corrected_20261001.md)选出的第 130 轮 `best.pt`。诊断只读模型和 VOC2007 test 数据，另存了阈值对照及探针结果；没有修改 checkpoint、数据集或训练配置。

## 具体症状

[原始总览，confidence ≥ 0.25](../../runs/yolov1/voc2007-resnet50-corrected-20261001/predictions/contact_sheet.jpg)中，`bottle` 图有 7 个真值框但只画 1 个预测框，`chair` 图画出 person 却没有 chair，`horse` 图漏掉马，`pottedplant`、`sheep`、`diningtable` 没有高于阈值的预测框，`train` 图画出 person/bicycle 而没有 train。[0.10 阈值对照总览](../../runs/yolov1/voc2007-resnet50-corrected-20261001/predictions_conf010/contact_sheet.jpg)可直接比较同一 20 张图片。

这 20 张图按固定种子 42 从每类真值中各取一张，且图片不重复。它们适合诊断具体失败模式，不代表完整 test 集的总体精度。

## 可复现的分层检查

1. **先核对输入与画框。** 对 `002231.jpg`，可视化入口的 RGB、双线性缩放、`/255` 张量与 test 数据读取器逐像素比较，最大差异为 **0**；标签解析张量也完全相同。推理和验证共同调用 `decode_predictions`，其输出是原图 `xyxy` 像素坐标；抽查的真值/预测左右图没有出现统一方向或固定比例的坐标平移。因此没有发现“只在画图时错位”的证据。共享解码器若有未发现的错误，仍可能同时影响验证和可视化。
2. **仅改变展示阈值。** 固定同一 checkpoint、图片、NMS IoU 0.45、每图最多 100 框；20 张图上用类别一致且 IoU≥0.5 的一对一匹配：

   | 展示 confidence | 真值 | 预测框 | 正确匹配 | 误检 | 漏检 | 样例 precision | 样例 recall |
   | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | 0.25 | 55 | 24 | 18 | 6 | 37 | 0.750 | 0.327 |
   | 0.10 | 55 | 70 | 31 | 39 | 24 | 0.443 | 0.564 |

   这说明原展示阈值太高，会隐藏不少候选；降阈值找回目标时也显著增加重复框和误检，不能把 0.10 的图误称为模型精度提升。[0.25 manifest](../../runs/yolov1/voc2007-resnet50-corrected-20261001/predictions/manifest.json)与[0.10 manifest](../../runs/yolov1/voc2007-resnet50-corrected-20261001/predictions_conf010/manifest.json)记录了每个框及分数。
3. **检查弱类是否只是被 0.25 隐藏。** [逐图阈值探针](../../runs/yolov1/voc2007-resnet50-corrected-20261001/visualization_threshold_probe.json)复用了同一次前向输出。`chair` 的本类最高分约 0.188，`horse` 约 0.205，降到 0.10 后各自出现与真值 IoU≥0.5 的框。`bottle` 在 0.25 时仅 1 个本类框且未达到 IoU 0.5；降到 0.10 后 7 个本类框中有 4 个真值获得 IoU≥0.5 的候选。`pottedplant` 的最高分约 0.149，但 0.10 下 8 个真值仍没有匹配框。`sheep`、`train`、`diningtable` 即使降到评估阈值 0.001，在原 NMS 设置下也匹配不到目标；它们是模型真正的分类/定位问题，而非单纯画图阈值问题。
4. **检查每图 100 框上限。** 在 0.25 下，20 张图总共只输出 24 框，100 框上限不可能解释这组可视化的漏检。降到 0.001 时，把上限从 100 提到 300，所选类别与真值 IoU≥0.5 的覆盖数均为 24/44；上限影响了低分候选的总数，但不是这组样例的主要原因。完整 test 平均 98.9 框/图，因此上限对全量 AP 的影响仍值得另做验证集对照，不能由 20 张样例外推。
5. **检查网格冲突与 NMS。** 所选 `bottle`、`car`、`chair`、`pottedplant` 图片的真值中心并未发生 7×7 同格冲突；它们的漏检不能直接归因于“同格只能监督一个目标”。在低阈值、最多 2,000 框的局部探针中，关闭 NMS 使 `bottle` 真值覆盖从 5/7 到 6/7，`sheep` 从 0/3 到 1/3；NMS 有局部影响，但关闭它会保留大量重复框，且仍无法解决 train/diningtable 等失败。

## 结合论文与参考实现判断

- [YOLOv1 原论文](https://arxiv.org/html/1506.02640)明确采用 7×7 网格、每格一组类别概率和两个框；原论文指出它难以处理成组小目标，且定位误差突出。这能解释架构上的上限，但不能替代上面的逐图实验：本次不少失败图没有同格真值冲突。
- 原论文用 VOC2007 **及 VOC2012 的 train/val** 训练约 135 轮；此 run 仅使用 VOC2007 train 的 **2,501 张图**。这是显著的数据量差异。当前 head 的第一层全连接已有约 **2.06 亿**个权重，有限样本下弱类学习与分数排序可能不足；这是有依据的风险判断，尚未通过只改变数据量的实验单独证明因果。
- [tanjeffreyz/yolo-v1 的绘图代码](https://raw.githubusercontent.com/tanjeffreyz/yolo-v1/main/utils.py)使用每格最高类别并以 0.2 为默认画图阈值；本项目对每个类别都计算 `class_score × box_confidence` 再做按类别 NMS，0.25 阈值且类别保留策略不同。两套图不能直接用“看起来框更多/更少”判断模型优劣。原论文的定义是逐类别得分，不能为了好看而悄悄改成最高类路径并声称仍是同一评估协议。
- [yakhyo/yolov1-pytorch 的推理代码](https://raw.githubusercontent.com/yakhyo/yolov1-pytorch/main/detect.py)同样先取每格最高类别，再分别按框置信度和类别乘积设阈值，随后做 NMS；入口示例还使用更高的阈值。它的[数据说明](https://github.com/yakhyo/yolov1-pytorch)同时列出 VOC2007 和 VOC2012 训练数据。因此该仓库的展示框数量及 README 指标也不能直接用来判断本项目的画框代码正确性或精度差距。
- [tanjeffreyz/yolo-v1 项目说明](https://github.com/tanjeffreyz/yolo-v1)也明确提到紧密排列和远处小目标的检测困难。对本 run，`bottle` 图存在同一类别的多个目标，`car` 图有 7 个较小目标；这些更接近定位/分数排序问题。[原论文](https://arxiv.org/html/1506.02640)说明 NMS 是消除重复框的后处理；本次 0.10 图中的大量框正说明降阈值会改变 precision/recall 权衡。

## 结论与下一步

本次可视化问题有**两层**：0.25 展示阈值让已有的低分正确候选不可见；模型本身在若干图上确实漏检或定位不准。上次只报 test mAP@0.50=0.5177 不足以说明实际展示效果，因为 AP 汇总了不同分数阈值下的排序表现，而固定阈值 0.25 的这组样例 recall 仅 0.327。test 逐类 AP@0.50 中 `pottedplant` 0.147、`bottle` 0.156、`chair` 0.211，也与可视化弱项一致。

当前应并排保留 0.25 和 0.10 图，清楚标明各自阈值和误检代价。若要提升模型而非只改变展示，应先在 **val** 上做不同阈值的 precision/recall、每类最佳工作点与 NMS/框上限对照，再固定评估协议；下一轮训练优先扩大 VOC 训练样本，同时保持独立 val/test，之后逐项比较更小的检测头或更适合密集目标的版本。不要用 test 样例挑选新配方后再把同一 test 分数当作独立验证。
