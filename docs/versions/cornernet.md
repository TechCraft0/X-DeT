# CornerNet

## 来源与定位

- 论文：Hei Law and Jia Deng, [CornerNet: Detecting Objects as Paired Keypoints](https://arxiv.org/abs/1808.01244), ECCV 2018。
- MMDetection 参考配置：[`cornernet_hourglass104_8xb6-210e-mstest_coco.py`](../../3dparty/mmdetection/configs/cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py)，本地参考 checkout 为 MMDetection `cfd5d3a985b0249de009b67d04f37263e11cdf3d`（Apache-2.0）。
- 公开检测权重：[CornerNet Hourglass-104 COCO checkpoint](https://download.openmmlab.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco/cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618-79b44c30.pth)。
- 本地下载文件大小 804,865,194 bytes，SHA256 `79b44c30d4bec042743b4d393ec0d9c2a078719829a89985b1cd2f7a01e7c9cb`；文件保存在用户 cache，不提交到 Git。
- 本项目在 [`x_yolo/models/cornernet/`](../../x_yolo/models/cornernet/) 中实现 Hourglass、corner pooling、目标、损失和解码，不依赖 MMDetection 或 MMCV。

## 方法与张量流

CornerNet 把框的左上角和右下角当作一对关键点。Hourglass-104 两个堆叠的 hourglass 各输出 stride-4、256 通道特征。每个 stack 的 head 分别预测：

- 左上与右下角类别热图：`[B, 20, H/4, W/4]`
- 两类角点的 offset：`[B, 2, H/4, W/4]`
- 左上与右下角各一个标量 associative embedding tag

Bi-directional corner pooling 为左上角沿向上/向左累计最大响应，为右下角沿向下/向右累计最大响应，以帮助网络结合边界上下文定位角点。Gaussian Focal Loss 监督两张热图；Smooth L1 监督整数网格内的角点 offset；Associative Embedding Loss 用 pull 项拉近同一物体的两个 tag、用 push 项分开不同物体的 tag。两层 hourglass 都接收监督。

推理对角点热图做局部峰值筛选，各取 top-100；按同类别、tag 距离不超过 0.5、左上到右下几何有效的组合配框。候选分数取两个角点分数的算术平均，随后运行 class-wise Gaussian Soft-NMS，最多保留 100 个框。

## 训练与评估

环境要求 Python 3.10+、PyTorch 与 TorchVision 版本匹配、Pillow、NumPy、PyYAML、Matplotlib。MMDetection 仅用作本地参考源码，运行时不需要安装。

```bash
mkdir -p ~/.cache/x-det/checkpoints
curl -fL https://download.openmmlab.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco/cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618-79b44c30.pth \
  -o ~/.cache/x-det/checkpoints/cornernet_hourglass104_coco.pth

python scripts/train_keypoint_detector.py \
  --recipe configs/cornernet/voc0712_hourglass104.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --pretrained ~/.cache/x-det/checkpoints/cornernet_hourglass104_coco.pth \
  --device cuda --output runs/cornernet/voc0712-hourglass104

python scripts/evaluate_keypoint_detector.py \
  --recipe configs/cornernet/voc0712_hourglass104.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --checkpoint runs/cornernet/voc0712-hourglass104/best.pt \
  --split test --device cuda \
  --output runs/cornernet/voc0712-hourglass104/test_metrics.json

python scripts/visualize_keypoint_predictions.py \
  --recipe configs/cornernet/voc0712_hourglass104.yaml \
  --dataset-root /path/to/yolo_voc2007_2012_trainmix \
  --checkpoint runs/cornernet/voc0712-hourglass104/best.pt \
  --split test --device cuda \
  --output-dir runs/cornernet/voc0712-hourglass104/test_visualizations
```

完整单卡训练参数见 [`configs/cornernet/voc0712_hourglass104.yaml`](../../configs/cornernet/voc0712_hourglass104.yaml)。输入 511×511、global batch 6、Adam、reference schedule 210 dataset passes；每 10 epoch 保存 checkpoint、每 5 epoch 在 VOC07 val 评估。可在 GPU 正式训练前用 `--max-steps 2 --epochs 1` 做有限 smoke，并按结果确定这张卡上的显存和吞吐。

训练曲线用 `scripts/plot_training_curves.py`；数据分布图用 `scripts/plot_dataset_analysis.py`；任意图片推理用 `scripts/infer_keypoint_detector.py`。

## 实现差异与验证状态

- 保留 Hourglass-104、corner pooling、左右上/右下角热图、offset、associative embedding、Gaussian Focal、Smooth L1 及 Gaussian Soft-NMS 的方法结构。
- recipe 通过公共 YOLO TXT loader 将输入拉伸到正方形，没有复刻参考的 RandomCenterCropPad、test-time multi-scale/flip；增强、单卡 batch 与按 batch 线性缩放的 Adam 学习率也据 VOC 任务调整。
- 真实 COCO checkpoint 已读取：1128 个张量与 Hourglass/head 结构匹配，占模型 state 的 99.999997%；角点类别通道按 VOC 类名映射。已完成 256×256 单 batch CPU 训练、验证和 best/last checkpoint 写入 smoke。它不代表 CUDA 显存、吞吐或检测精度；GPU smoke 与正式训练仍待 FCOS 结束后验证。
- CornerNet 规模明显高于 ResNet-18 CenterNet。先报告一次真实显存与 step 时间，再决定是否按完整 recipe 运行；不能把 CPU smoke 当作 GPU 训练可行性或精度证据。
