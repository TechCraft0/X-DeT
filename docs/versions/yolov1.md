# YOLOv1：从网格检测到可运行的训练路径

## 来源与定位

- 论文：Joseph Redmon 等，*You Only Look Once: Unified, Real-Time Object Detection*，CVPR 2016。
- 论文：[arXiv 页面](https://arxiv.org/abs/1506.02640)，[CVPR 论文页](https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)。
- 参考实现：[pjreddie/darknet 的 yolov1.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg)。该 cfg 是工程配置，包含 BN、local/detection 层等设置，不等同于论文网络；本目录以论文描述为教学基线。
- 用户补充的实现参考：[yakhyo/yolov1-pytorch](https://github.com/yakhyo/yolov1-pytorch) 和 [tanjeffreyz/yolo-v1](https://github.com/tanjeffreyz/yolo-v1)。
- 状态：YOLOv1 风格 ResNet-50 迁移变体已完成 VOC2007 的 135 轮训练、独立 test 评估和曲线/样例可视化；详见[修正后完整训练报告](../reports/voc2007_yolov1_resnet50_corrected_20261001.md)。后续 ConvNeXt Small 14×14 + VOC2007/2012 train 数据扩展在 VOC2007 test 上达到 mAP50 `0.67084`、mAP50:95 `0.28683`；详见[曝光量匹配实验报告](../experiments/yolov1/voc0712_convnext_small_grid14_exposurematched_20261005.md)。这些主干/检测头组合是教学型工程变体，论文主干的完整复现与目标后端验证仍是独立工作。

## 核心思想

YOLOv1 把检测写成一次网络前向：输入整图，输出 S×S 网格。每个网格预测 B 个框及其置信度，并给出一组该网格的条件类别分数。论文基线使用 S=7、B=2、输入 448×448；输出通道为 `5B+C`。对本数据的 C=2，原始张量为 `[N,7,7,12]`。

相对两阶段检测器，这种设计用整图上下文直接回归框和类别，推理流程紧凑；代价是每个网格只有一个类别分布，密集目标尤其容易冲突。两个框预测器不能解决同一网格中多个目标的类别/位置监督冲突。

## 结构和张量流

```text
RGB [N,3,448,448]
  -> 24 个卷积层和 max-pooling / stride 卷积
  -> 特征 [N,1024,7,7]
  -> FC 4096 + LeakyReLU + Dropout(0.5)
  -> 588 个数 (C=2)
  -> raw grid [N,7,7,12]
  -> 解码 + class-aware NMS
  -> 每张图的 xyxy 像素框、class_id、confidence
```

每格的 12 个数按 `[box_1(5), box_2(5), class_scores(2)]` 排列。框为 `(x_cell_offset, y_cell_offset, sqrt_width, sqrt_height, confidence)`；宽高以整图归一化值的平方根形式回归。类别分数是线性输出，不经 softmax。解码时每类置信分数按 `class_score × box_confidence` 计算。

损失默认使用论文中的平方误差形式：坐标项乘 `λ_coord=5`，未负责框的置信度项乘 `λ_noobj=0.5`；类别项只在有目标网格计算。每格两个预测框与该格真值分别算 IoU，IoU 较高者负责坐标和目标置信度回归，置信度目标为该预测框与真值的 IoU。代码以批次大小归一化各项，便于记录；这不改变同一 batch 内各项的相对权重。为可控研究定位目标，`YoloV1Loss` 另提供默认关闭的 `lambda_giou=0`；打开后将 GIoU 损失加到坐标项，属于工程扩展，不是原始 YOLOv1 损失。

## 文件地图

- `x_yolo/models/yolov1/model.py`：24 卷积层、两层全连接输出头。
- `x_yolo/models/yolov1/resnet50_transfer.py`：显式独立的 ResNet-50 迁移学习变体；支持按训练阶段冻结/解冻 ImageNet backbone，再接 YOLOv1 风格检测头，维持相同 raw grid 输出契约。
- `x_yolo/models/yolov1/torchvision_conv_transfer.py`：TorchVision 预训练分类主干加紧凑卷积检测头的实验变体，支持多个主干家族；默认输出 7×7，也可在配方中显式选 14×14。卷积头替代了 YOLOv1 的大规模全连接检测头，不属于论文原版网络。
- `x_yolo/models/factory.py`：只列出当前实际实现的模型选择，不做自动插件发现。
- `x_yolo/models/yolov1/loss.py`：每格一个真值目标、责任框按 IoU 选择、论文平方误差损失。
- `x_yolo/models/yolov1/postprocess.py`：网格解码、类别分数、纯 PyTorch 类别感知 NMS。
- `x_yolo/data/dataset.py`：YOLO TXT 读取、成对校验、方形缩放及训练增强。
- `x_yolo/training/trainer.py`：训练循环、梯度累积、恢复、验证和运行记录。
- `x_yolo/evaluation/metrics.py`：按类别计算 AP@0.5 和 AP@[0.5:0.95]。
- `scripts/`：数据检查、训练、验证、单图推理命令。
- `configs/datasets/yolo_final_20k.yaml`：类名和 split 相对路径；本机根目录通过 CLI 传入。
- `configs/yolov1/train.yaml`：本次 YOLOv1 训练配方。

## 数据约定与当前数据

输入标注为每图一个同名 TXT 文件，每行 `class_id x_center y_center width height`，坐标和宽高均相对原图归一化。类别顺序为 `person`, `car`。图像统一 RGB，然后直接缩放到 448×448；这会改变非正方形图像的长宽比，符合本基线的简单预处理选择，评估预测再映射回原图像素坐标。数据读取只读用户数据，检查脚本会报告错配、越界、非有限值和类别错误。

已检查的本地数据集是 `yolo_final_20k`：训练 18,000 张/227,692 框，验证 2,000 张/24,500 框；类别分别为训练 person 78,767、car 148,925，验证 person 8,573、car 15,927。两 split 的图片和标签一一配对，扫描到的标签行有效。原数据集 YAML 中有一条旧机器绝对路径；本项目配置不读取或修改该 YAML。

**数据与 YOLOv1 网格不匹配。** 训练集中 16,408 张图、验证集 1,817 张图至少有两个标注中心落在同一 7×7 网格；训练集有 128,892 个“同格额外框”，验证有 13,726 个。训练目标采用一个明确的适配：每格选择面积最大的目标，面积相同时保留 TXT 中较早一行；被忽略目标数写入每轮指标。验证 AP 仍使用全部原始标签。该策略让训练目标确定且可复现，但不能消除 YOLOv1 对密集场景的结构性限制，结果不应被解释为完整数据监督下的原始论文复现。

## 配方、训练和检查点

配方主要按论文报告：448 输入、7×7、每格两个框、SGD、momentum 0.9、weight decay 0.0005、初始学习率 0.01、约 135 epoch 和 batch 64 的有效批次，并使用色彩、尺度、平移扰动。论文还先用 ImageNet 分类任务预训练 backbone，再以 224 输入预训练后切到 448；本项目不自动下载权重，当前 recipe 的 checkpoint 为 `null`，即随机初始化。这项差异和“每格最大面积目标”的数据适配都会显著影响训练结果。

### 两个 PyTorch 仓库的参考边界

- `yakhyo/yolov1-pytorch` 的 backbone 是 Darknet 风格 20 个卷积层，检测 head 再接 4 个卷积层和 FC；实现把 head 的所有输出都过 sigmoid，并通过本地 ImageNet backbone checkpoint 加载预训练。可参考其模块拆分和数据/评估入口；输出激活与权重加载约定不直接等同本项目的 raw linear 输出。
- `tanjeffreyz/yolo-v1` 提供两条路径：默认讨论的迁移学习路径用冻结的 torchvision ResNet50，并接检测头；这改变了 YOLOv1 的 Darknet 主干。该仓库的 `loss.py` 把负责框的置信度目标写成 1，而论文目标是预测框与真值框的 IoU，和其 README 的文字定义也不完全一致。其 `plot.py` 主要保存测试图的预测框可视化，不是训练曲线绘图。
- 两个仓库都显示 ImageNet 表征有助于训练，尤其 tanjeffreyz 路径直接冻结预训练 backbone。不同数据集上的 README mAP 不能直接拿来当本地 `yolo_final_20k` 的预期结果。
- 为在标准数据上检查训练链路，另实现了 `yolov1_resnet50_transfer`：使用 torchvision ResNet-50 ImageNet 权重（需通过命令行传入本机文件），按 ImageNet 均值/方差归一化，再接 4 层卷积和 FC 检测头。它借鉴 tanjeffreyz 的迁移学习路径，保留项目的 YOLOv1 输出、损失和评估代码；ResNet 主干和输入归一化是明确的工程变体，不能称为论文原版 YOLOv1。该路径不自动联网下载权重。
- 首次 VOC 135 轮结果使用整程冻结的 ResNet-50，精度很低。修正后的完整训练使用 `voc2007_resnet50_corrected.yaml`：前 5 轮冻结主干，从第 6 轮起解冻全部 ResNet 参数；Adam，检测头学习率 `1e-4`、backbone 学习率 `1e-5`，并在第 75/105 轮降学习率。每 10 轮保存一次含优化器状态的 `last.pt`，验证 mAP@[0.50:0.95] 刷新时保存 `best.pt`，末轮强制保存。实测结果见[完整报告](../reports/voc2007_yolov1_resnet50_corrected_20261001.md)。

VOC2007 数据可用 `scripts/prepare_voc.py` 从官方 devkit XML 转成 YOLO TXT。默认省略 `difficult=1` 实例并将数量写入 `conversion_report.json`；因此项目 AP 与官方 VOC devkit AP 有定义差异，不能直接比较。转换器会复制官方 train/val/test split，不修改源 XML 或图片；`evaluate_yolov1.py --split test` 用于单独测试集评估。

### TorchVision 主干实验

本机 TorchVision 0.25.0 已核对以下模型在 448×448 输入时的最后特征形状，均为 `[N,C,14,14]`；卷积检测头将其变为 `[N,7,7,30]`（VOC 20 类）。这只是接口/张量检查，不代表所有主干都已完整训练。

| `backbone_name` | 特征通道 `C` | 预训练方式 |
| --- | ---: | --- |
| `resnet18`, `resnet34` | 512 | TorchVision 权重枚举或匹配的本地 checkpoint |
| `resnet50` | 2048 | TorchVision 权重枚举或匹配的本地 checkpoint |
| `resnext50_32x4d` | 2048 | TorchVision 权重枚举或匹配的本地 checkpoint |
| `regnet_y_400mf`, `regnet_y_800mf` | 440, 784 | TorchVision 权重枚举或匹配的本地 checkpoint |
| `efficientnet_b0` | 1280 | TorchVision 权重枚举或匹配的本地 checkpoint |
| `efficientnet_v2_s` | 1280 | TorchVision 权重枚举或匹配的本地 checkpoint |
| `convnext_tiny`, `convnext_small` | 768 | TorchVision 权重枚举或匹配的本地 checkpoint |
| `mobilenet_v3_large` | 960 | TorchVision 权重枚举或匹配的本地 checkpoint |

`resnet30` 不是 TorchVision 提供的模型名。选择权重时，`model.backbone_checkpoint` 指向已存在的匹配 state dict，或者 `model.backbone_weights: DEFAULT` 明确请求 TorchVision 官方权重；二者只能选一个。`DEFAULT` 若未缓存，会由 TorchVision 下载，下载行为由 recipe 明示而不是模块导入时触发。此变体只采用官方权重的 RGB 均值/方差归一化，检测输入仍直接缩放为 448×448，不使用分类模型的中心裁剪预处理。更多模型和权重枚举见 [TorchVision 官方文档](https://docs.pytorch.org/vision/stable/models.html)。

首个受控对照使用[ResNet50 卷积头配方](../../configs/yolov1/voc2007_resnet50_convhead.yaml)：沿用上一轮 ResNet50 预训练文件、VOC2007 train/val、损失、优化器、冻结时长、batch、增强和评估设置，仅把原先包含约 2.06 亿权重的首个全连接层所在检测头换成约 1,063 万参数的卷积头。30 轮结果和曲线见[训练策略报告](../reports/voc2007_training_strategy_20261001.md)。[ConvNeXt Tiny 配方](../../configs/yolov1/voc2007_convnext_tiny_convhead.yaml)与[ConvNeXt Small 配方](../../configs/yolov1/voc2007_convnext_small_convhead.yaml)使用同一检测头和 VOC 配方，分别比较两种 TorchVision 规模。RegNet 和 ResNeXt 的 30 轮筛选以及 ConvNeXt Small 的完整训练结果见[骨干与训练策略实验记录](../experiments/yolov1/convnext_small_voc2007_20261002.md)。EfficientNetV2-S 目前已通过张量接口检查，尚无本项目检测精度结论；更换骨干时应分别另建配方和运行目录，避免混用 checkpoint。

为验证 7×7 网格的同格目标限制，卷积检测头新增显式 14×14 选项：448 输入下 ConvNeXt 特征为 14×14，7×7 头用首层 stride 2，14×14 头用 stride 1。标签分配和损失从配方读取同一个 `grid_size`，解码依据模型输出网格大小恢复坐标。YOLOv1 的每格两个框和一组类别分数仍保留，但把论文的 7×7 改成 14×14 是工程变体；默认仍为7×7。当前配方见[ConvNeXt Small 14×14 配置](../../configs/yolov1/voc2007_convnext_small_grid14.yaml)，与[7×7 配置](../../configs/yolov1/voc2007_convnext_small_convhead.yaml)分别运行、保存检查点。

14×14 与精度日程匹配的 7×7 完整对照（135 epochs）、同 test split 的逐类指标和 validation 置信度校准见[ConvNeXt Small 网格对照实验记录](../experiments/yolov1/convnext_small_grid14_voc2007_20261004.md)。受控 test 结果为 7×7 `0.62818 / 0.25931`、14×14 `0.64769 / 0.26100`（mAP50 / mAP50:95）；高 IoU 指标仅小幅变化，仍不应视作完成部署验证。

针对 14×14 trainmix 配方中“框大致包含物体、边界偏松”的现象，YOLOv1 风格损失增加了可选的 GIoU 辅助项；默认系数为 0，因此既有配方目标不变。ConvNeXt Small VOC2007+2012 train 对照中只启用 `lambda_giou=1`，test mAP50/mAP50:95 从 coord10 的 `0.66679 / 0.28911` 提高到 `0.68754 / 0.31846`。这是原论文平方误差定位项之外的工程扩展，不是原始 YOLOv1 损失复现；训练曲线、同图框对照和限制见 [GIoU 定位实验记录](../experiments/yolov1/voc0712_convnext_small_grid14_giou_aux_20261005.md)。VOC test 曾参与前序模型选择，因此这组数据是受控项目对比，不是新的盲测结论。

```bash
python scripts/inspect_dataset.py \
  --config configs/datasets/yolo_final_20k.yaml \
  --root /path/to/yolo_final_20k

python scripts/train_yolov1.py \
  --dataset-root /path/to/yolo_final_20k \
  --device cuda

python scripts/evaluate_yolov1.py \
  --dataset-root /path/to/yolo_final_20k \
  --checkpoint runs/yolov1/<run>/best.pt --device cuda \
  --output runs/yolov1/<run>/final_validation.json

python scripts/infer_yolov1.py path/to/image.jpg \
  --checkpoint runs/yolov1/<run>/best.pt --device cuda
```

VOC2007 转换及迁移学习变体示例：

```bash
python scripts/prepare_voc.py \
  --voc-root /path/to/VOCdevkit/VOC2007 \
  --output-root /path/to/yolo_voc2007

python scripts/train_yolov1.py \
  --recipe configs/yolov1/voc2007_resnet50_corrected.yaml \
  --dataset-root /path/to/yolo_voc2007 \
  --backbone-checkpoint /path/to/resnet50-0676ba61.pth \
  --device cuda
```

短程连通性运行可使用 `--epochs 1 --max-steps 2`，这只是流程检查，不代表模型收敛。恢复训练示例：

```bash
python scripts/train_yolov1.py --dataset-root /path/to/yolo_final_20k \
  --resume runs/yolov1/<run>/last.pt --device cuda
```

每个 run 会保存配方快照、数据配置快照、`run_config.yaml`、逐 epoch 的 `metrics.jsonl`、按配方 `checkpoint_interval_epochs` 周期写入的 `last.pt` 和按验证 mAP50:95 选择的 `best.pt`。最终 epoch 总会写 `last.pt`；配置更长的保存间隔可减少大型模型的磁盘写入，但中断后最多会重做一个间隔内的 epoch。实验报告记录数据名、split/class 映射、输入预处理、种子、运行环境、设备、配置、代码 revision（若仓库有 Git revision）和验证指标。数据、权重和运行目录不纳入源码目录。

`--micro-batch-size` 可以覆盖单次前向的物理 batch；它必须整除配方里的 `effective_batch_size`。训练器相应改变梯度累积次数，并在 `run_config.yaml` 记录两种 batch 与累积步数。改变物理 batch 可能影响含 BatchNorm 的骨干训练统计，因此性能对照应明确记录这项设置。

`--precision bf16` 在支持 BF16 的 CUDA 设备上使用自动混合精度运行模型前向，模型输出转回 FP32 后再计算 YOLOv1 坐标/IoU/平方误差损失；默认仍为 FP32。验证仍用 FP32。BF16 是单独记录的训练设置，不能把续训前后曲线差异全部归因于学习率。

训练完成后用 `python scripts/plot_training_curves.py --run-dir runs/yolov1/<run>` 生成 `training_curves.png` 与可放大查看的 `training_curves.svg`，曲线包括总损失及分项、验证集逐类 AP@0.50/AP@[0.50:0.95]、两个 mAP、学习率和每轮未参与网格监督的标注数。类别名从指标中动态读取。AP 图以零为下界，并按观测最大值自适应上界；上界会显示在图标题中，低 AP 阶段也能看清变化。若一次 run 分阶段续训且保留了 `stage1_run_config.yaml`，辅助图会标出阶段续训位置。验证指标仅在实际执行验证的 epoch 有记录。

## 指标和结果含义

评估采用连续插值面积 AP，IoU 阈值为 0.50 到 0.95（步长 0.05），逐类报告 AP50/AP50:95，并对有标注的类别取均值得到 mAP。验证框来自所有 YOLO TXT 标签；训练网格碰撞的忽略规则不应用于评估。通用预测结果契约为原图 `xyxy` 像素坐标、类别整数 ID 和 confidence。模型张量比较/ONNX/TensorRT/RKNN 尚属后续独立工作，不能用检测 AP 代替后端数值一致性报告。

## 后续版本和自定义方法

新增论文版本先在 `docs/versions/<version>.md` 写清论文与官方实现来源、网络/特征融合/检测头、标签分配、损失、增强、推理的差异，再在独立 `x_yolo/models/<version>/` 中实现版本特有路径。只有实测多个版本行为一致的部分才放入公共 data/training/evaluation 编排。保留各版原始配方；同数据、相同 split 下评估，结果同时记录 recipe、输入尺寸、预处理和数据碰撞策略。自定义模块或损失应以单独的版本/实验配置进入，避免用 `if version == ...` 把各论文差异埋进同一个模型类。

## 当前验证状态

整程冻结的 ResNet-50 基线报告见[VOC2007 首轮训练与测试报告](../reports/voc2007_yolov1_resnet50_20260930.md)：test mAP@0.50 为 0.07844，mAP@[0.50:0.95] 为 0.02935。随后修正增强，从干净 ImageNet 权重训练了 135 轮；最佳第 130 轮 checkpoint 在 held-out VOC2007 test 达到 mAP@0.50 **0.51765**、mAP@[0.50:0.95] **0.21241**，并已保存训练曲线、数据分析与 20 张 test 真值/预测对照图。详见[修正后完整报告](../reports/voc2007_yolov1_resnet50_corrected_20261001.md)。VOC `difficult` 目标被转换器省略，项目 AP 定义与官方 VOC devkit 有差异。

固定 confidence 0.25 的展示图漏检较多，降至 0.10 可找回部分目标但误检显著增加；若干类别在更低阈值下仍定位或分类失败。证据及文献对照见[可视化问题诊断](../reports/voc2007_yolov1_visualization_diagnosis_20261001.md)。

后续微调 run 于第 114 轮后停止：`last.pt` 是第 110 轮，按验证 mAP@[0.50:0.95] 选出的 `best.pt` 是第 90 轮；该 run 没有进行 held-out test。训练 loss 从第 1 轮 12.12 降至第 114 轮 1.81，验证 mAP@[0.50:0.95] 在第 90 轮达到 0.1365，说明增加少量 epoch 不是主要修复办法。停训分析见[ResNet-50 微调停训诊断](../reports/voc2007_yolov1_resnet50_finetune_stopped_20260930.md)。

诊断发现并修复了一项影响训练标签正确性的仿射增强错误：旧代码将图像绕左上角缩放，却将框绕图像中心缩放。训练中启用随机缩放和平移，因此两者会错位。新代码把 PIL 的逆向采样矩阵改为与框变换相同的“绕中心缩放再平移”，并新增确定性回归测试。旧 checkpoint 保留用于审计；修正后的完整训练从干净的 ImageNet backbone 开始，没有从旧 `last.pt` 续训。

用户自己的 `yolo_final_20k` 不在本轮继续训练；对应 run 在 33 个完整 epoch 后暂停，权重和日志保留。其部分曲线保存在 `runs/yolov1/baseline-epoch1-20260929/training_curves_partial.png` 与 `.svg`。大型数据和实验产物不提交到源码仓库。
