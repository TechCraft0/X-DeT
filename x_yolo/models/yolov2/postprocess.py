"""Decode YOLOv2 anchor logits into the shared xyxy-pixel detection contract."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from x_yolo.models.detection_utils import class_aware_nms


def decode_predictions(
    output: torch.Tensor,
    original_sizes: list[tuple[int, int]],
    anchors: torch.Tensor | Sequence[Sequence[float]],
    confidence_threshold: float = 0.001,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 100,
) -> list[dict[str, torch.Tensor]]:
    """Decode `[N,S,S,A,5+C]` raw logits using YOLOv2's anchor equations."""
    if output.ndim != 5:
        raise ValueError(f"Expected [N,S,S,A,5+C], got {tuple(output.shape)}")
    batch_size, grid_h, grid_w, num_anchors, channels = output.shape
    if grid_h != grid_w or len(original_sizes) != batch_size or channels < 6:
        raise ValueError("Expected a square grid, one image size per batch item and at least one class")
    if max_detections < 1:
        raise ValueError("max_detections must be at least 1")
    if not 0 <= confidence_threshold <= 1 or not 0 <= nms_iou_threshold <= 1:
        raise ValueError("confidence and NMS thresholds must be in [0,1]")
    anchor_tensor = torch.as_tensor(anchors, dtype=output.dtype, device=output.device)
    if anchor_tensor.shape != (num_anchors, 2) or not torch.isfinite(anchor_tensor).all() or (anchor_tensor <= 0).any():
        raise ValueError("anchors must be finite positive [A,2] pairs matching the prediction tensor")

    rows = torch.arange(grid_h, device=output.device, dtype=output.dtype).view(1, grid_h, 1, 1)
    columns = torch.arange(grid_w, device=output.device, dtype=output.dtype).view(1, 1, grid_w, 1)
    centers = torch.stack(
        ((columns + output[..., 0].sigmoid()) / grid_w, (rows + output[..., 1].sigmoid()) / grid_h),
        dim=-1,
    )
    sizes = output[..., 2:4].clamp(-10, 10).exp() * anchor_tensor.view(1, 1, 1, num_anchors, 2) / grid_w
    normalized_xywh = torch.cat((centers, sizes), dim=-1)
    half = normalized_xywh[..., 2:4] / 2
    boxes = torch.cat((normalized_xywh[..., :2] - half, normalized_xywh[..., :2] + half), dim=-1)
    boxes[..., 0::2].clamp_(0, 1)
    boxes[..., 1::2].clamp_(0, 1)
    class_probabilities = output[..., 5:].softmax(dim=-1)
    scores = class_probabilities * output[..., 4].sigmoid().unsqueeze(-1)
    results: list[dict[str, torch.Tensor]] = []

    for batch_index, (image_width, image_height) in enumerate(original_sizes):
        pixel_boxes = boxes[batch_index].reshape(-1, 4) * output.new_tensor(
            [image_width, image_height, image_width, image_height]
        )
        class_scores = scores[batch_index].reshape(-1, channels - 5)
        candidate_boxes, candidate_scores, candidate_labels = [], [], []
        for class_id in range(channels - 5):
            selected = torch.where(class_scores[:, class_id] >= confidence_threshold)[0]
            if selected.numel() == 0:
                continue
            candidate_boxes.append(pixel_boxes[selected])
            candidate_scores.append(class_scores[selected, class_id])
            candidate_labels.append(torch.full_like(selected, class_id))
        if not candidate_boxes:
            results.append({
                "boxes": output.new_zeros((0, 4)),
                "scores": output.new_zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.long, device=output.device),
            })
            continue
        all_boxes = torch.cat(candidate_boxes)
        all_scores = torch.cat(candidate_scores)
        all_labels = torch.cat(candidate_labels)
        keep = class_aware_nms(all_boxes, all_scores, all_labels, nms_iou_threshold)[:max_detections]
        results.append({"boxes": all_boxes[keep], "scores": all_scores[keep], "labels": all_labels[keep]})
    return results
