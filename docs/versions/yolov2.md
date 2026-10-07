# YOLOv2 / YOLO9000 检测器

## 来源与范围

- 论文：Joseph Redmon、Ali Farhadi，[*YOLO9000: Better, Faster, Stronger*](https://arxiv.org/abs/1612.08242)，CVPR 2017。
- 主要工程对照：[Darknet VOC 配置](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg)。本实现使用其中的 VOC anchors 和主要检测损失权重作为参考。
- 代码交叉参考：[longcw/yolo2-pytorch](https://github.com/longcw/yolo2-pytorch)、[miladlink/YoloV2](https://github.com/miladlink/YoloV2)。前者 README 提醒其依赖较旧 PyTorch；两者用于理解模块组织，不作为运行依赖或精度依据。
- 阅读辅助：[Docsaid YOLOv2 notes](https://docsaid.org/en/papers/object-detection/yolov2/)。算法定义仍以论文和 Darknet 配置为准。

这里实现的是 YOLOv2 单阶段检测器的教学型 PyTorch 路径，不包括 YOLO9000 的 WordTree 和检测/分类联合训练。模型拓扑以 Darknet-19 和 passthrough 特征为核心；训练入口可显式加载官方 Darknet-19 ImageNet 分类权重的前 18 个特征卷积块，检测专属层仍随机初始化。权重不会自动下载。ImageNet 分类预训练不等于官方 YOLOv2 检测器权重，也不能单独视为完整论文复现。

## YOLOv2 的主要变化

YOLOv1 在每个网格预测固定数量的框，类别概率与框置信度分开。YOLOv2 引入 anchor boxes，用训练框的宽高形状先验来预测每个网格的多个候选框；中心位置经 sigmoid 限制在负责网格内，宽高用指数变换乘以 anchor。论文还提出 IoU 距离的维度聚类、Darknet-19、passthrough 高分辨率特征，以及分类预训练后的高分辨率微调等方法。

YOLO9000 论文还讨论 WordTree 和联合训练。它们服务于把检测数据与分类数据合并训练，不属于本仓库当前的 VOC 检测器实现。

## 结构和张量流

默认输入为 `[N,3,416,416]`，像素先按 RGB 除以 255。卷积主体使用 `Conv2d → BatchNorm2d → LeakyReLU(0.1)`：

| 阶段 | 输出形状（416 输入） | 作用 |
|---|---|---|
| Darknet-19 前半段 | `[N,512,26,26]` | 产生 stride 16 的 route 特征 |
| route 1×1 降维 | `[N,64,26,26]` | 减少 passthrough 通道 |
| reorg / `pixel_unshuffle(2)` | `[N,256,13,13]` | 把局部 2×2 空间信息搬到通道维 |
| Darknet-19 后半段 | `[N,1024,13,13]` | 产生 stride 32 的深层特征 |
| 拼接与融合卷积 | `[N,1024,13,13]` | 融合两种分辨率的特征 |
| 检测输出 | `[N,13,13,5,5+C]` | 每个格点 5 个 anchor；通道为 `tx,ty,tw,th,obj,class_logits` |

输入高度和宽度必须是 32 的倍数。输出是原始 logits，不在 `forward` 中执行阈值筛选或 NMS。代码入口见 [model.py](../../x_yolo/models/yolov2/model.py)。

默认 VOC anchor 按一个 stride-32 网格单元计：`(1.3221,1.73145)`, `(3.19275,4.00944)`, `(5.05587,8.09892)`, `(9.47112,4.84053)`, `(11.2364,10.0071)`。例如输入 416 时网格为 13×13，每个网格单元为 32 像素。

## 训练目标与标签分配

对某个 cell 和 anchor，解码为：

```text
b_x = (sigmoid(t_x) + cell_x) / S
b_y = (sigmoid(t_y) + cell_y) / S
b_w = exp(t_w) * anchor_w / S
b_h = exp(t_h) * anchor_h / S
P(object) = sigmoid(t_o)
P(class | object) = softmax(class_logits)
```

训练时每个真值框放入其中心所在 cell，并选择与真值宽高 IoU 最大的 anchor。正样本框坐标、objectness 和类别采用 Darknet 风格平方误差；正样本 objectness 目标是停止梯度的预测框/真值框 IoU（对应配置的 `rescore`）；其他 anchor 若与任一真值框 IoU 超过阈值，则从负 objectness 项中忽略。

实现细节及边界：

- 同一个 cell 内如果多个框竞争同一 anchor 槽，保留 anchor 形状 IoU 较高的标签，其余计入 `ignored_ground_truths`；这些真值仍参与负样本 ignore 计算。这是固定张量目标格式下的显式冲突策略，不声称与 Darknet 的所有边界行为完全相同。
- 默认 anchors 直接采用 Darknet VOC 配置，不执行论文中的 IoU 距离 k-means。新数据集应统计框宽高并重新聚类/评估 anchors 后再训练。
- 位置与宽高使用原始平方误差；当前实现没有额外的 IoU/GIoU 辅助项，也没有按目标面积加权。
- 损失实现见 [loss.py](../../x_yolo/models/yolov2/loss.py)，推理解码/NMS 见 [postprocess.py](../../x_yolo/models/yolov2/postprocess.py)。输出检测框契约是原图像素 `xyxy`、类别 ID 和 `objectness × class_probability` 分数。

## 文件与命令

- 模型：`x_yolo/models/yolov2/model.py`
- 目标分配与损失：`x_yolo/models/yolov2/loss.py`
- 解码和 NMS：`x_yolo/models/yolov2/postprocess.py`
- 训练配置：`configs/yolov2/voc0712.yaml`
- 训练：

```bash
mkdir -p weights/yolov2
curl -fL https://pjreddie.com/media/files/darknet19_448.weights \
  -o weights/yolov2/darknet19_448.weights

python scripts/train_yolov2.py \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --pretrained-darknet weights/yolov2/darknet19_448.weights \
  --device cuda
```

权重来源是[官方 Darknet ImageNet 权重页](https://pjreddie.com/darknet/imagenet/)。该文件是 448 输入的分类 checkpoint；加载器会复制 Darknet-19 前 18 个卷积块的卷积核和 BatchNorm 状态，并跳过 1000 类分类卷积。分类输出层与检测专属层不迁移。SHA256 为 `77bd0b33f92522a97d6667c5d6cb118d4928bec40f21872f5b4965231ac2167b`；`weights/` 已加入 `.gitignore`。VOC 数据应为配置中的 YOLO TXT 结构：`images/train`, `labels/train`, `images/val`, `labels/val`，类别顺序与 `configs/datasets/voc0712_trainmix.yaml` 一致。当前配置按 416 输入训练、有效 batch 64、每 10 个 epoch 保存可恢复的 `last.pt`，评估周期内的最佳权重保存为 `best.pt`。配置中的训练轮数与学习率是本项目的起始方案，不是声称逐项复刻 Darknet 的训练调度。

- 评估：

```bash
python scripts/evaluate_yolov2.py \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --checkpoint runs/yolov2/<run>/best.pt \
  --split val --device cuda
```

- 单图推理：

```bash
python scripts/infer_yolov2.py image.jpg \
  --checkpoint runs/yolov2/<run>/best.pt --device cuda
```

- 训练曲线、数据集分布图和真值/预测对照图：

```bash
python scripts/plot_training_curves.py --run-dir runs/yolov2/<run>
python scripts/plot_dataset_analysis.py \
  --config configs/datasets/voc0712_trainmix.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --output-prefix runs/yolov2/<run>/dataset_analysis --grid-size 13
python scripts/visualize_yolov2_predictions.py \
  --recipe configs/yolov2/voc0712.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --checkpoint runs/yolov2/<run>/best.pt \
  --output-dir runs/yolov2/<run>/predictions --split val --device cuda
```

数据分析图中的网格碰撞统计只表示中心落在同一网格，不等同于 YOLOv2 的 cell/anchor 分配冲突；版本文档和训练指标会单独说明实际标签分配冲突数。

训练入口支持 `--max-steps` 做有界的流程检查；它不是收敛或精度验证。所有命令都要求由用户明确触发，不会在导入模块时下载数据/权重或占用 GPU。

## 验证状态

已完成 VOC 2007 train + VOC 2012 trainval 上的 165 epoch 训练，并在 VOC 2007 val 与 VOC 2007 test 上完整评估。后续固定其余设置、以 `1e-5` 续训 30 epochs，没有超过原验证最佳值；该对照见[续训实验记录](../experiments/yolov2/voc0712_low_lr_continuation_20261006.md)。当前推荐权重仍为 epoch165，在 VOC2007 test 得到 mAP50 `0.694`、mAP50:95 `0.394`；YOLOv1 对比、逐类差异、训练曲线、数据分析和预测可视化见[完整训练报告](../reports/yolov2_voc0712_vs_yolov1_20261005.md)。`runs/` 下的权重和图表是本机实验产物并被 Git 忽略；推理用原 run 的 `best_inference.pt`，续训恢复点为 epoch195 的 `last.pt`。基础实现与有界 smoke 记录见[实现验证记录](../reports/yolov2_implementation_smoke_20261005.md)。

WordTree 联合训练、动态多尺度训练和 ONNX/设备后端部署尚不包含在本版本中。
