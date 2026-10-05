# X-YOLO

面向学习与研究的 YOLO 系列代码实现。项目从 YOLOv1 的模型、标签与损失、训练和评估纵向流程开始，优先说明各论文版本的真实差异。

## YOLOv1 第一个实验

当前适配的数据集配置是 `configs/datasets/yolo_final_20k.yaml`（person/car，YOLO TXT）。传入数据集根目录即可检查和运行：

```bash
python scripts/inspect_dataset.py --config configs/datasets/yolo_final_20k.yaml --root /path/to/yolo_final_20k
python scripts/train_yolov1.py --dataset-root /path/to/yolo_final_20k --device cuda
python scripts/evaluate_yolov1.py --dataset-root /path/to/yolo_final_20k --checkpoint runs/yolov1/<run>/best.pt --device cuda
python scripts/infer_yolov1.py image.jpg --checkpoint runs/yolov1/<run>/best.pt --device cuda
python scripts/plot_training_curves.py --run-dir runs/yolov1/<run>
```

训练配置和数据约束见 [YOLOv1 版本说明](docs/versions/yolov1.md)，首轮训练实测见 [训练与验证报告](docs/reports/yolov1_yolo_final_20k_epoch1.md)。训练会保留每次运行的配置、指标和 checkpoint；不要把用户数据复制到仓库。

已完成的 VOC2007 ResNet-50 迁移实验见[135 轮训练与测试报告](docs/reports/voc2007_yolov1_resnet50_corrected_20261001.md)，其中包含训练曲线、数据集分析、逐类 AP 和 20 张真值/预测对照图。该模型是 YOLOv1 风格工程变体，报告列明了与论文及 VOC 官方评估的差异。

推理图片存在漏检、误检及展示阈值影响；逐图实验和论文对照见[可视化问题诊断](docs/reports/voc2007_yolov1_visualization_diagnosis_20261001.md)。

## YOLOv2

仓库包含 Darknet-19 + passthrough 的 YOLOv2 检测器、anchor 标签分配、YOLOv2 损失和独立训练/评估/推理入口。模型返回 anchor logits，和 YOLOv1 使用不同的标签与损失路径。实现范围、论文差异、VOC anchors 和运行命令见 [YOLOv2 版本说明](docs/versions/yolov2.md)。默认配置使用 VOC 先验，但随机初始化且没有完整论文训练策略；请以版本说明中的限制和实际训练报告为准。
