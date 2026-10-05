"""Decode the original YOLOv1 grid output to pixel-space detections."""
from __future__ import annotations

import torch
from x_yolo.models.detection_utils import box_iou_xyxy, class_aware_nms


def decode_predictions(
    output: torch.Tensor,
    original_sizes: list[tuple[int, int]],
    confidence_threshold: float = 0.001,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 100,
) -> list[dict[str, torch.Tensor]]:
    """Convert raw `[N,S,S,5B+C]` predictions to xyxy pixel detections.

    `original_sizes` is a list of `(width, height)`. YOLOv1 emits linear class
    scores and confidence values; class score is their product, as in the paper.
    Width and height are emitted in square-root space and squared at inference.
    """
    if output.ndim != 4:
        raise ValueError(f"Expected [N,S,S,channels], got {tuple(output.shape)}")
    batch_size, grid_h, grid_w, channels = output.shape
    if grid_h != grid_w or len(original_sizes) != batch_size:
        raise ValueError("Expected a square grid and one original image size per prediction")
    if channels < 11:
        raise ValueError("YOLOv1 output needs at least two box predictors and one class")
    boxes_per_cell = 2
    num_classes = channels - boxes_per_cell * 5
    # YOLOv1 has two box tuples followed by one class distribution per cell:
    # `[box1(5), box2(5), classes(C)]`, so the output is not reshaped into
    # two independent `(5+C)` records.
    box_predictions = output[..., : boxes_per_cell * 5].reshape(
        batch_size, grid_h, grid_w, boxes_per_cell, 5
    )
    class_predictions = output[..., boxes_per_cell * 5 :]
    rows = torch.arange(grid_h, device=output.device, dtype=output.dtype).view(1, grid_h, 1, 1)
    cols = torch.arange(grid_w, device=output.device, dtype=output.dtype).view(1, 1, grid_w, 1)
    results: list[dict[str, torch.Tensor]] = []

    for batch_index, (image_width, image_height) in enumerate(original_sizes):
        one = box_predictions[batch_index]
        center_x = (cols[0] + one[..., 0]) / grid_w
        center_y = (rows[0] + one[..., 1]) / grid_h
        width = one[..., 2].clamp_min(0).square()
        height = one[..., 3].clamp_min(0).square()
        normalized_boxes = torch.stack(
            (center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2), dim=-1
        ).reshape(-1, 4)
        normalized_boxes[:, 0::2].clamp_(0, 1)
        normalized_boxes[:, 1::2].clamp_(0, 1)
        pixel_scale = output.new_tensor([image_width, image_height, image_width, image_height])
        boxes = normalized_boxes * pixel_scale
        confidence = one[..., 4].clamp(0, 1).reshape(-1, 1)
        class_scores = class_predictions[batch_index].unsqueeze(-2).expand(
            grid_h, grid_w, boxes_per_cell, num_classes
        ).reshape(-1, num_classes)
        scores_by_class = class_scores * confidence
        candidate_boxes, candidate_scores, candidate_labels = [], [], []
        for class_id in range(num_classes):
            class_scores_one = scores_by_class[:, class_id]
            selected = torch.where(class_scores_one >= confidence_threshold)[0]
            if selected.numel() == 0:
                continue
            candidate_boxes.append(boxes[selected])
            candidate_scores.append(class_scores_one[selected])
            candidate_labels.append(torch.full_like(selected, class_id))
        if not candidate_boxes:
            results.append(
                {
                    "boxes": output.new_zeros((0, 4)),
                    "scores": output.new_zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.long, device=output.device),
                }
            )
            continue
        all_boxes = torch.cat(candidate_boxes)
        all_scores = torch.cat(candidate_scores)
        all_labels = torch.cat(candidate_labels)
        keep = class_aware_nms(all_boxes, all_scores, all_labels, nms_iou_threshold)[:max_detections]
        results.append({"boxes": all_boxes[keep], "scores": all_scores[keep], "labels": all_labels[keep]})
    return results
