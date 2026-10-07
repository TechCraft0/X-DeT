"""Dataset-level average precision for normalized detection outputs."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from x_yolo.models.yolov1.postprocess import decode_predictions


def _iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=1)
    area_one = (box[2:] - box[:2]).clamp_min(0).prod()
    area_many = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=1)
    return intersection / (area_one + area_many - intersection).clamp_min(1e-9)


def _integral_ap(recall: torch.Tensor, precision: torch.Tensor) -> float:
    if recall.numel() == 0:
        return 0.0
    mrec = torch.cat((recall.new_zeros(1), recall, recall.new_ones(1)))
    mpre = torch.cat((precision.new_zeros(1), precision, precision.new_zeros(1)))
    mpre = torch.cummax(mpre.flip(0), dim=0).values.flip(0)
    changes = torch.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]).sum().item())


def average_precision_at_iou(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float,
) -> dict[int, float]:
    """Continuous integral AP with confidence-ordered, one-to-one matching.

    IoUs are calculated per image in vectorized NumPy arrays. This preserves
    greedy confidence-order matching while avoiding millions of tiny Torch
    operations during multi-threshold VOC evaluation.
    """
    class_ap: dict[int, float] = {}
    for class_id in range(num_classes):
        total_ground_truths = 0
        score_parts: list[np.ndarray] = []
        best_iou_parts: list[np.ndarray] = []
        matched_gt_parts: list[np.ndarray] = []
        gt_key_offset = 0
        for image_index, target in enumerate(targets):
            target_mask = target["labels"] == class_id
            ground_truth = target["boxes"][target_mask].detach().cpu().numpy().astype(np.float32, copy=False)
            total_ground_truths += len(ground_truth)
            pred = predictions[image_index]
            pred_mask = pred["labels"] == class_id
            boxes = pred["boxes"][pred_mask].detach().cpu().numpy().astype(np.float32, copy=False)
            scores = pred["scores"][pred_mask].detach().cpu().numpy().astype(np.float32, copy=False)
            score_parts.append(scores)
            if len(boxes) and len(ground_truth):
                top_left = np.maximum(boxes[:, None, :2], ground_truth[None, :, :2])
                bottom_right = np.minimum(boxes[:, None, 2:], ground_truth[None, :, 2:])
                intersection = np.prod(np.maximum(bottom_right - top_left, 0), axis=2)
                box_area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
                gt_area = np.prod(np.maximum(ground_truth[:, 2:] - ground_truth[:, :2], 0), axis=1)
                union = box_area[:, None] + gt_area[None, :] - intersection
                overlaps = intersection / np.maximum(union, 1e-9)
                best_gt = overlaps.argmax(axis=1)
                best_iou_parts.append(overlaps[np.arange(len(boxes)), best_gt])
                matched_gt_parts.append(gt_key_offset + best_gt)
            else:
                # An image without a GT for this class contributes only false
                # positives; the dummy key is never eligible at IoU >= 0.5.
                best_iou_parts.append(np.zeros(len(boxes), dtype=np.float32))
                matched_gt_parts.append(np.full(len(boxes), -1, dtype=np.int64))
            gt_key_offset += len(ground_truth)
        if total_ground_truths == 0:
            continue
        scores = np.concatenate(score_parts)
        best_ious = np.concatenate(best_iou_parts)
        matched_gt = np.concatenate(matched_gt_parts)
        order = np.argsort(-scores, kind="stable")
        scores = scores[order]
        best_ious = best_ious[order]
        matched_gt = matched_gt[order]

        eligible = np.flatnonzero(best_ious >= iou_threshold)
        true_positive = np.zeros(len(scores), dtype=np.float32)
        if len(eligible):
            # For a GT, only its highest-confidence eligible prediction is TP.
            # Unique keys keep the first occurrence in the stable score order.
            _, first_for_gt = np.unique(matched_gt[eligible], return_index=True)
            true_positive[eligible[first_for_gt]] = 1.0
        false_positive = 1.0 - true_positive
        true_positive = np.cumsum(true_positive, dtype=np.float32)
        false_positive = np.cumsum(false_positive, dtype=np.float32)
        recall = true_positive / total_ground_truths
        precision = true_positive / np.maximum(true_positive + false_positive, 1e-9)
        recall = np.concatenate((np.zeros(1, dtype=np.float32), recall, np.ones(1, dtype=np.float32)))
        precision = np.concatenate((np.zeros(1, dtype=np.float32), precision, np.zeros(1, dtype=np.float32)))
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        recall_changes = np.flatnonzero(recall[1:] != recall[:-1])
        ap = np.sum(
            (recall[recall_changes + 1] - recall[recall_changes]) * precision[recall_changes + 1],
            dtype=np.float32,
        )
        class_ap[class_id] = float(ap)
    return class_ap


def evaluate_predictions(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    class_names: Sequence[str],
) -> dict[str, object]:
    thresholds = [0.5 + 0.05 * index for index in range(10)]
    per_threshold = {
        f"AP@{threshold:.2f}": average_precision_at_iou(predictions, targets, len(class_names), threshold)
        for threshold in thresholds
    }
    present_classes = [
        class_id for class_id in range(len(class_names))
        if any(bool((target["labels"] == class_id).any()) for target in targets)
    ]
    ap50 = per_threshold["AP@0.50"]
    ap50_95_values = [per_threshold[f"AP@{threshold:.2f}"] for threshold in thresholds]
    ap50_95 = {
        class_id: sum(table.get(class_id, 0.0) for table in ap50_95_values) / len(thresholds)
        for class_id in present_classes
    }
    return {
        "AP50": {class_names[class_id]: ap50.get(class_id, 0.0) for class_id in present_classes},
        "mAP50": sum(ap50.get(class_id, 0.0) for class_id in present_classes) / max(len(present_classes), 1),
        "AP50_95": {class_names[class_id]: ap50_95[class_id] for class_id in present_classes},
        "mAP50_95": sum(ap50_95.values()) / max(len(present_classes), 1),
    }


def _targets_to_pixels(boxes: torch.Tensor, image_size: tuple[int, int]) -> dict[str, torch.Tensor]:
    width, height = image_size
    if boxes.numel() == 0:
        return {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}
    class_ids = boxes[:, 0].to(torch.long)
    center_x, center_y, box_width, box_height = boxes[:, 1:].unbind(dim=1)
    pixel_boxes = torch.stack(
        (
            (center_x - box_width / 2) * width,
            (center_y - box_height / 2) * height,
            (center_x + box_width / 2) * width,
            (center_y + box_height / 2) * height,
        ),
        dim=1,
    )
    return {"boxes": pixel_boxes, "labels": class_ids}


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    class_names: Sequence[str],
    device: torch.device,
    confidence_threshold: float = 0.001,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 100,
    decoder: Callable[..., list[dict[str, torch.Tensor]]] = decode_predictions,
) -> dict[str, object]:
    if max_detections < 1:
        raise ValueError("max_detections must be at least 1")
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    for images, box_lists, _names, original_sizes in data_loader:
        outputs = model(images.to(device))
        batch_predictions = decoder(
            outputs,
            list(original_sizes),
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
        )
        # AP is a CPU-side dataset metric. Move each batch once here instead
        # of synchronizing/copying individual detections from CUDA inside the
        # per-class, per-threshold matching loops below.
        predictions.extend(
            {name: value.detach().cpu() for name, value in prediction.items()}
            for prediction in batch_predictions
        )
        targets.extend(_targets_to_pixels(boxes, size) for boxes, size in zip(box_lists, original_sizes))
    metrics = evaluate_predictions(predictions, targets, class_names)
    metrics["images"] = len(targets)
    metrics["detections"] = sum(int(prediction["boxes"].shape[0]) for prediction in predictions)
    metrics["max_detections"] = max_detections
    return metrics
