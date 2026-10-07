"""Dataset-level average precision for normalized detection outputs."""
from __future__ import annotations

from collections.abc import Callable, Sequence

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
    """Continuous integral AP, matching each ground-truth box at most once."""
    class_ap: dict[int, float] = {}
    for class_id in range(num_classes):
        gt_by_image: dict[int, torch.Tensor] = {}
        total_ground_truths = 0
        detections: list[tuple[float, int, torch.Tensor]] = []
        for image_index, target in enumerate(targets):
            mask = target["labels"] == class_id
            boxes = target["boxes"][mask].detach().cpu()
            gt_by_image[image_index] = boxes
            total_ground_truths += boxes.shape[0]
            pred = predictions[image_index]
            pred_mask = pred["labels"] == class_id
            for box, score in zip(pred["boxes"][pred_mask], pred["scores"][pred_mask]):
                detections.append((float(score.detach().cpu()), image_index, box.detach().cpu()))
        if total_ground_truths == 0:
            continue
        detections.sort(key=lambda item: item[0], reverse=True)
        matched = {image_index: torch.zeros(boxes.shape[0], dtype=torch.bool) for image_index, boxes in gt_by_image.items()}
        true_positive = torch.zeros(len(detections), dtype=torch.float32)
        false_positive = torch.zeros(len(detections), dtype=torch.float32)
        for detection_index, (_, image_index, box) in enumerate(detections):
            ground_truth = gt_by_image[image_index]
            if ground_truth.shape[0] == 0:
                false_positive[detection_index] = 1
                continue
            overlaps = _iou_one_to_many(box, ground_truth)
            best_iou, best_index = overlaps.max(dim=0)
            if best_iou >= iou_threshold and not matched[image_index][best_index]:
                true_positive[detection_index] = 1
                matched[image_index][best_index] = True
            else:
                false_positive[detection_index] = 1
        tp = true_positive.cumsum(0)
        fp = false_positive.cumsum(0)
        recall = tp / total_ground_truths
        precision = tp / (tp + fp).clamp_min(1e-9)
        class_ap[class_id] = _integral_ap(recall, precision)
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
