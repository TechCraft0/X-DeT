# YOLOv1 首轮训练与验证报告

## 本轮范围

- 数据：`yolo_final_20k`，训练 18,000 张，验证 2,000 张，类别顺序 `person`, `car`。
- 训练：2026-09-29 23:55 至 2026-09-30 00:04（约 8 分 55 秒）；1 个完整 epoch、4,500 个 micro-batch step。recipe 计划 135 epoch，本轮显式覆盖为 1 epoch，用于跑通全量训练与验证链路。
- 设备：NVIDIA GeForce RTX 4060 Ti，16 GB 显存；Python 3.10.12、PyTorch 2.10.0+cu128、CUDA 12.8。
- 配方：448×448、S=7、B=2、micro batch 4、梯度累计 16 次（effective batch 64）、SGD、momentum 0.9、weight decay 0.0005、seed 42。
- 初始化：随机初始化。论文使用 ImageNet 分类预训练；本轮没有预训练权重，所以这是代码/流程基线，不能视为论文复现或已收敛模型。
- 数据碰撞：增强后的本 epoch 目标分配统计 125,108 个同格额外标注未参与该格监督。原始 split 静态扫描统计 train 128,892、val 13,726。验证 AP 使用全部验证标签。

## 验证结果

| 指标 | person | car | mean |
|---|---:|---:|---:|
| AP@0.50 | 0.00000682 | 0.00007797 | 0.00004239 |
| AP@[0.50:0.95] | 0.00000083 | 0.00001144 | 0.00000613 |

验证集为 2,000 张图；在 confidence threshold 0.001 和 NMS IoU 0.45 下，最终解码输出达到每图 100 个检测框上限，共 200,000 个检测。这与近零 AP 一起说明当前 checkpoint 的预测质量很差，不能用于实际检测。单图 CLI 在 confidence 0.25 下执行成功，返回空检测列表。

## 检查记录

- `python -m pytest -q tests/test_yolov1.py`：3 passed。
- `python -m compileall -q x_yolo scripts`：通过。
- 两步 CUDA 端到端短跑：成功完成 forward/backward、验证集 AP、checkpoint 保存；AP 为 0，符合未训练模型的短跑状态。
- 一 epoch 全训练集运行：4,500 step，loss 各项和 collision 忽略计数正常，完整验证集指标成功计算。
- 独立 `evaluate_yolov1.py` 命令：完成并将报告写入 `runs/yolov1/baseline-epoch1-20260929/final_validation.json`。
- `infer_yolov1.py` 单图命令：成功加载 checkpoint 并输出标准 JSON 检测契约。
- `openspec validate yolov1-comparison-foundation --strict`：通过。

## 可追溯产物

- 配置和环境：`runs/yolov1/baseline-epoch1-20260929/run_config.yaml`
- 逐 epoch 记录：`runs/yolov1/baseline-epoch1-20260929/metrics.jsonl`
- 完整验证报告：`runs/yolov1/baseline-epoch1-20260929/final_validation.json`
- 权重：`runs/yolov1/baseline-epoch1-20260929/best.pt`、`last.pt`

当前 mAP 低且检测数达到上限；在讨论版本间准确率对比前，应补上预训练初始化、训练到完整计划或明确一致的预算，并检查置信度/NMS 与标注网格冲突的影响。保留此报告中的 epoch 和 recipe 限制，不能把本轮写成“训练收敛”或“复现论文结果”。
