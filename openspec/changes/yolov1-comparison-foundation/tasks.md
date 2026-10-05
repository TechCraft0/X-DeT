## 1. Confirm the local data and runtime contract

- [x] 1.1 Inspect `yolo_final_20k` and document its YOLO TXT contract, 18,000/2,000 train/val pairs, person/car class order, label validity, and 7×7 collision counts without changing source data.
- [x] 1.2 Inspect the runtime: Python 3.10.12, PyTorch 2.10.0+cu128, Pillow 11.2.1, PyYAML 6.0.3, NumPy 1.26.4, and torchvision 0.25.0+cu128 are available; use PyTorch and avoid adding torchvision as a required dependency.
- [x] 1.3 Define the dataset configuration fields and validation errors for paths, classes, splits, and YOLO TXT labels.

## 2. Implement the YOLOv1 data and model path

- [x] 2.1 Implement the configured YOLO TXT dataset reader and the preprocessing needed by the selected YOLOv1 recipe.
- [x] 2.2 Implement the YOLOv1 model with documented tensor shapes and explicit raw output semantics.
- [x] 2.3 Implement the YOLOv1 target construction and loss with formula references and readable shape checks.
- [x] 2.4 Implement YOLOv1 decoding and postprocessing, documenting coordinate and threshold conventions.

## 3. Add recipe-driven training and experiment records

- [x] 3.1 Add the YOLOv1 training recipe configuration, recording its paper or official-source provenance and any adaptation to the user's data.
- [x] 3.2 Implement the shared training entry point for dataset selection, optimizer steps, checkpoint recovery, seed, device, and configured output directory.
- [x] 3.3 Save a per-run configuration snapshot and metadata for dataset split, class mapping, code revision, environment, preprocessing, evaluation settings, and metrics.
- [x] 3.4 Add a post-training plot command for loss components, learning rate, ignored targets, and validation AP curves.

## 4. Add shared evaluation and inference entry points

- [x] 4.1 Define and implement the common detection result contract using `xyxy` pixel coordinates, class ID, and confidence.
- [x] 4.2 Implement dataset evaluation for AP@0.5 and AP@[0.5:0.95], including per-class and aggregate results.
- [x] 4.3 Implement YOLOv1 image inference and root-level commands for training, evaluation, and inference.
- [x] 4.4 Keep dataset evaluation results separate from any future PyTorch-versus-backend parity report.

## 5. Document and validate the first vertical slice

- [x] 5.1 Write the YOLOv1 version document covering source, ideas, tensor flow, code map, data and recipe configuration, run commands, differences, and validation status.
- [x] 5.2 Document how to add a later paper version or custom method by isolating changed components and reusing only demonstrated shared orchestration.
- [x] 5.3 Run unit and end-to-end checks, then train and validate using the requested local data; report the actual duration, settings, metrics, and limitations.
- [ ] 5.4 Original `yolo_final_20k` full run: paused at 33 complete epochs after the user redirected validation to VOC2007; preserve its checkpoints and logs for a later resume.

## 6. Validate a transfer-learning path on public VOC data

- [x] 6.1 Download and verify the official VOC2007 train/val and annotated test archives; retain the source archives and untouched extracted tree outside the repository.
- [x] 6.2 Convert VOC XML annotations into a derived YOLO TXT dataset, retain official splits, record omitted difficult objects, and scan all image/label pairs.
- [x] 6.3 Add an explicit ResNet-50 transfer variant with local-only pretrained checkpoint loading, shared YOLOv1 loss/evaluator, recipe selection, and selectable validation/test split.
- [x] 6.4 Add conversion coverage and verify the 20-class forward/loss/backward path plus dynamic per-class training plots.
- [x] 6.5 Complete the 135-epoch VOC run, evaluate the held-out test split once, generate PNG/SVG curves, and write a final report with remaining limitations. Training/evaluation completed; VOC test mAP@0.50=0.07844 and mAP@[0.50:0.95]=0.02935. Pipeline works, model accuracy is not acceptable yet. See `docs/reports/voc2007_yolov1_resnet50_20260930.md`.

## 7. Diagnose the fine-tuned VOC run and correct augmentation

- [x] 7.1 Stop the fine-tuning process and preserve its epoch metrics, best checkpoint, last periodic checkpoint, and partial training curves without overwriting the earlier frozen-backbone run.
- [x] 7.2 Compare the image affine transform with the YOLO box transform; fix their center-scaling mismatch and add a deterministic alignment regression test.
- [x] 7.3 Record whether the result is epoch-limited, summarize the partial validation evidence, and mark held-out test/new training as not run in `docs/reports/voc2007_yolov1_resnet50_finetune_stopped_20260930.md`.
- [x] 7.4 After a clean restart from ImageNet weights, pass fixed-geometry visualization and small-sample overfit checks, then run corrected VOC training and held-out evaluation. Completed 135 epochs; best epoch 130 reaches VOC2007 test mAP@0.50=0.51765 and mAP@[0.50:0.95]=0.21241. See `docs/reports/voc2007_yolov1_resnet50_corrected_20261001.md`.
- [x] 7.5 Save training curves, VOC dataset analysis, per-class test AP, and deterministic ground-truth/prediction examples in the corrected run directory.
