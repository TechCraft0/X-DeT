# VOC0712：低学习率延长训练

**日期：** 2026-10-06　 **目的：** 检查 epoch165 后的低学习率阶段是否仍能提高验证集定位指标。

## 假设与对照

原始 165 epoch 训练的最佳验证 mAP50:95 出现在最后一轮，且 epoch150 至165 仍有上升，因此假设固定在 `1e-5` 再训练 30 epochs 可能继续细化定位。

唯一改变的训练变量是总 epoch 数：165 → 195。模型、输入 416、VOC 数据、Darknet-19 初始化、SGD 状态、有效 batch 64、增强、损失、随机种子及验证配置保持不变。续训从原 `last.pt`（epoch165，含 optimizer 状态）恢复，学习率里程碑 `[146,164]` 让 epoch166–195 始终使用 `1e-5`。

配置：[`voc0712_continue_lr1e5.yaml`](../../../configs/yolov2/voc0712_continue_lr1e5.yaml)。续训输出：`runs/yolov2/voc0712-darknet19-pretrained-continue-lr1e5-30ep-20261006/`。

复跑命令（将 `DATASET_ROOT` 设为相同 VOC trainmix 数据目录）：

```bash
python scripts/train_yolov2.py \
  --dataset-root "$DATASET_ROOT" \
  --recipe configs/yolov2/voc0712_continue_lr1e5.yaml \
  --resume runs/yolov2/voc0712-darknet19-pretrained-20261005/last.pt \
  --output runs/yolov2/voc0712-darknet19-pretrained-continue-lr1e5-30ep-20261006 \
  --device cuda:0 --precision bf16 --micro-batch-size 4
```

## 结果

| Epoch | mAP50 | mAP50:95 | Train loss |
|---:|---:|---:|---:|
| 165（基线最佳） | 0.68529 | **0.39206** | 0.40767 |
| 170 | 0.68488 | 0.39005 | 0.40755 |
| 175 | 0.68805 | 0.39163 | 0.39969 |
| 180 | **0.68823** | 0.39190 | 0.40031 |
| 185 | 0.68509 | 0.39052 | 0.40010 |
| 190 | 0.68477 | 0.39062 | 0.39281 |
| 195 | 0.68472 | 0.38963 | **0.39004** |

续训期间最高 mAP50:95 是 epoch180 的 `0.39190`，比基线低 `0.00017`；训练损失最终降低到 `0.39004`，但没有转化为更好的高 IoU 验证表现。mAP50 在 epoch175/180 短暂升高，之后回落。结果表明**仅延长当前低 LR 阶段没有提高模型的最佳定位指标**。

新 run 的 `best.pt` 保持为从原 run 复制来的 epoch165 checkpoint；`last.pt` 是 epoch195，供需要时继续恢复。全局最佳模型与原始最佳模型参数完全相同，因此沿用原来的推理权重及其 VOC2007 test 结果：mAP50 `0.69365`、mAP50:95 `0.39408`。没有用 test split 选择本次模型。

## 产物

- [完整 1–195 epoch 训练与验证曲线](../../../runs/yolov2/voc0712-darknet19-pretrained-continue-lr1e5-30ep-20261006/training_curves.png)
- [续训最佳权重（epoch165，与原 checkpoint 相同）](../../../runs/yolov2/voc0712-darknet19-pretrained-continue-lr1e5-30ep-20261006/best.pt)
- [续训最终恢复点（epoch195）](../../../runs/yolov2/voc0712-darknet19-pretrained-continue-lr1e5-30ep-20261006/last.pt)

checkpoint 和曲线都在本机 `runs/` 中并被 Git 忽略。

## 结论与下一步

停止重复延长 `1e-5` 阶段。若继续追求定位精度，下一次应单独测试更高输入分辨率微调（例如 512×512），保留原损失与数据增强，按 validation mAP50:95 选模；VOC2007 test 只在选定候选后作最终报告。若提高分辨率仍无收益，再另开实验评估 VOC train split 上重新聚类 anchors 或加入 IoU 辅助框损失，并明确标注为工程变体。
